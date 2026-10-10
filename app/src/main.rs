use std::process::ExitCode;

use clap::Parser;
use moe_jav::api::http::HttpTransport;
use moe_jav::api::jev::JevClient;
use moe_jav::api::llm::LmStudioClient;
use moe_jav::cli::{Cli, run};
use moe_jav::config::Config;
use moe_jav::error::Result;

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "moe_jav=info".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    match start(Cli::parse()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

async fn start(cli: Cli) -> Result<()> {
    let config = Config::from_env()?;
    let http = HttpTransport::new(config.http_timeout)?;
    let llm = LmStudioClient::new(http.clone(), config.llm);
    let jev = JevClient::new(http, config.jev);
    run(cli, &llm, &jev, &mut std::io::stdout().lock()).await
}
