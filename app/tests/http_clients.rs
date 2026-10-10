//! Integration tests for the real HTTP clients against a local mock server.

use std::time::Duration;

use moe_jav::api::http::{HttpTransport, MAX_RESPONSE_BYTES};
use moe_jav::api::jev::{DecisionModel, JevClient, JevRequest, Question};
use moe_jav::api::llm::{JsonPrompt, LlmClient, LmStudioClient};
use moe_jav::config::{ApiKey, BaseUrl, JevConfig, LlmConfig};
use moe_jav::error::AppError;
use serde_json::json;
use std::collections::BTreeMap;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn transport() -> HttpTransport {
    HttpTransport::new(Duration::from_secs(5))
        .unwrap()
        .with_retries(2, Duration::ZERO)
}

fn local(server: &MockServer) -> BaseUrl {
    BaseUrl::new(&format!("{}/v1", server.uri())).unwrap()
}

fn jev(server: &MockServer) -> JevClient {
    JevClient::new(
        transport(),
        JevConfig {
            base_url: local(server),
            model: "jev-latest".into(),
            api_key: ApiKey::new("test-key").unwrap(),
        },
    )
}

fn llm(server: &MockServer) -> LmStudioClient {
    LmStudioClient::new(
        transport(),
        LlmConfig {
            base_url: local(server),
            model: "google/gemma-4-e4b".into(),
            reasoning_effort: Some("none".into()),
            max_tokens: 1024,
        },
    )
}

fn noul_request() -> JevRequest {
    JevRequest {
        state: json!({"application": "I smoke."}),
        questions: BTreeMap::from([(
            "smoker".to_owned(),
            Question::Noul {
                instructions: "Is the applicant a smoker?".into(),
                criteria: None,
            },
        )]),
    }
}

#[tokio::test]
async fn jev_sends_bearer_and_parses_answers() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .and(header("authorization", "Bearer test-key"))
        .and(body_partial_json(json!({
            "model": "jev-latest",
            "questions": {"smoker": {"type": "noul"}}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "jev-1.13.0",
            "answers": {"smoker": {"type": "noul", "noul": 0.97}},
            "usage": {"input_tokens": 10, "output_tokens": 2}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let resp = jev(&server).decide(&noul_request()).await.unwrap();
    assert_eq!(resp.model, "jev-1.13.0");
    assert!((resp.noul("smoker").unwrap() - 0.97).abs() < 1e-9);
}

#[tokio::test]
async fn jev_missing_answer_is_bad_response() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({"model": "j", "answers": {}})),
        )
        .mount(&server)
        .await;
    let err = jev(&server).decide(&noul_request()).await.unwrap_err();
    assert!(matches!(err, AppError::BadResponse { .. }), "{err}");
}

#[tokio::test]
async fn jev_malformed_body_is_bad_response() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"unexpected": 1})))
        .mount(&server)
        .await;
    assert!(matches!(
        jev(&server).decide(&noul_request()).await,
        Err(AppError::BadResponse { .. })
    ));
}

#[tokio::test]
async fn client_errors_are_not_retried() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(401).set_body_string("bad key"))
        .expect(1)
        .mount(&server)
        .await;
    match jev(&server).decide(&noul_request()).await {
        Err(AppError::Status { status, body, .. }) => {
            assert_eq!(status, 401);
            assert_eq!(body, "bad key");
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[tokio::test]
async fn server_errors_are_retried_then_succeed() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(503))
        .up_to_n_times(2)
        .expect(2)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "j", "answers": {"smoker": {"type": "noul", "noul": 0.5}}
        })))
        .expect(1)
        .mount(&server)
        .await;
    assert!(jev(&server).decide(&noul_request()).await.is_ok());
}

#[tokio::test]
async fn retries_are_bounded() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(500))
        .expect(3)
        .mount(&server)
        .await;
    assert!(matches!(
        jev(&server).decide(&noul_request()).await,
        Err(AppError::Status { status: 500, .. })
    ));
}

#[tokio::test]
async fn oversized_response_is_rejected() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string("x".repeat(MAX_RESPONSE_BYTES + 1)),
        )
        .mount(&server)
        .await;
    let err = jev(&server).decide(&noul_request()).await.unwrap_err();
    assert!(err.to_string().contains("exceeds"), "{err}");
}

#[tokio::test]
async fn non_json_response_is_rejected() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string("<html>"))
        .mount(&server)
        .await;
    let err = jev(&server).decide(&noul_request()).await.unwrap_err();
    assert!(err.to_string().contains("invalid JSON"), "{err}");
}

#[tokio::test]
async fn connection_failure_is_http_error() {
    // Reserve a free port, then release it so nothing is listening there.
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let client = JevClient::new(
        HttpTransport::new(Duration::from_secs(5))
            .unwrap()
            .with_retries(0, Duration::ZERO),
        JevConfig {
            base_url: BaseUrl::new(&format!("http://127.0.0.1:{port}/v1")).unwrap(),
            model: "jev-latest".into(),
            api_key: ApiKey::new("k").unwrap(),
        },
    );
    assert!(matches!(
        client.decide(&noul_request()).await,
        Err(AppError::Http { .. })
    ));
}

#[tokio::test]
async fn llm_sends_schema_and_extracts_content() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({
            "model": "google/gemma-4-e4b",
            "reasoning_effort": "none",
            "response_format": {"type": "json_schema", "json_schema": {"name": "route", "strict": true}}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"message": {"content": "{\"expert\":\"medical\",\"confidence\":0.7}"}}]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let v = llm(&server)
        .complete_json(&JsonPrompt {
            system: "s",
            user: "u",
            schema_name: "route",
            schema: json!({"type": "object"}),
        })
        .await
        .unwrap();
    assert_eq!(v["expert"], "medical");
}

fn route_prompt() -> JsonPrompt<'static> {
    JsonPrompt {
        system: "s",
        user: "u",
        schema_name: "route",
        schema: json!({"type": "object"}),
    }
}

#[tokio::test]
async fn llm_sends_token_cap() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(body_partial_json(json!({"max_tokens": 1024})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]
        })))
        .expect(1)
        .mount(&server)
        .await;
    llm(&server).complete_json(&route_prompt()).await.unwrap();
}

#[tokio::test]
async fn llm_truncated_with_effort_retries_without_effort() {
    let server = MockServer::start().await;
    // Runaway thinking with reasoning_effort=none: truncated at the cap.
    Mock::given(method("POST"))
        .and(body_partial_json(json!({"reasoning_effort": "none"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"finish_reason": "length", "message": {"content": ""}}]
        })))
        .expect(1)
        .mount(&server)
        .await;
    // Retry omits reasoning_effort and succeeds.
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"finish_reason": "stop", "message": {"content": "{\"ok\":true}"}}]
        })))
        .expect(1)
        .mount(&server)
        .await;
    let v = llm(&server).complete_json(&route_prompt()).await.unwrap();
    assert_eq!(v["ok"], true);
    let requests = server.received_requests().await.unwrap();
    let second: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert!(second.get("reasoning_effort").is_none());
}

#[tokio::test]
async fn llm_truncated_twice_is_bad_response() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"finish_reason": "length", "message": {"content": ""}}]
        })))
        .expect(2)
        .mount(&server)
        .await;
    let err = llm(&server)
        .complete_json(&route_prompt())
        .await
        .unwrap_err();
    assert!(err.to_string().contains("max_tokens"), "{err}");
}
