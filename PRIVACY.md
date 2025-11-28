# Privacy Policy

Last updated: 2025-10-29

This Privacy Policy explains how the MCP Learning Path Generator (the "App") collects, uses, and shares information when you use the App. The App is provided for demonstration and personal productivity purposes.

## Summary

- What we collect: OAuth tokens for YouTube when you opt in via the OAuth flow, and basic non-personal usage telemetry in your browser session (only while the app runs locally or in your deployed session).
- Why we collect it: To allow the App to perform actions on your behalf (for example, create playlists on your YouTube account) and to generate learning paths using external content.
- Sharing: We do not sell or share your OAuth tokens. Tokens are stored locally in the deployment environment's filesystem (`tokens/youtube_token.json`) and are used only to call Google APIs on your behalf.

## Data we collect

1. OAuth tokens (access and refresh tokens)
   - Purpose: To call YouTube APIs to create or manage playlists and other account-scoped operations when you explicitly enable YouTube OAuth.
   - Storage: Tokens are cached to a local file (`tokens/youtube_token.json`) in the deployed environment. Tokens are not sent to third-party analytics or stored in any external database by the App.

2. Environment variables and configuration
   - Purpose: The App reads configuration values (API keys, MCP endpoints) from environment variables or a local `.env` file. These values are used to connect to external services and are not transmitted to the App author.

3. Agent/debug logs
   - Purpose: The App may produce debug logs if `DEBUG_DIRECT_TOOLS` is enabled; logs can include truncated request/response previews for debugging. Do not enable debug in production with real secrets.

## How we use your information

- The App uses your OAuth token to perform YouTube operations that you request through the UI.
- The App may forward your token to the configured MCP or tool endpoints only when necessary to complete the requested operation.

## Third-party services

- Google (YouTube): OAuth authentication and API calls. The App uses standard Google OAuth flows and Google's APIs to act on your behalf.
- Composio or other configured MCP endpoints: The App may call configured MCP endpoints to discover and invoke tools.

## Security

- Tokens are stored on disk in the deployment environment. Treat the deployment environment as you would treat any credentials store; protect it from unauthorized access.
- Do not commit `youtube_client_secrets.json` or token files to a public repository. These files are listed in `.gitignore`.

## Your choices and rights

- You can revoke the App's access in your Google Account's security settings (https://myaccount.google.com/permissions).
- To remove locally stored tokens, delete the file at `tokens/youtube_token.json` in the deployment environment.

## Contact

If you have questions about this Privacy Policy, please contact the application owner or the repository maintainer.

---

This privacy policy is provided for verification purposes for OAuth consent/verification flows. It is intentionally minimal; if Google requests additional details during verification, the App owner should expand this document to satisfy the requested requirements (data retention periods, contact email, detailed data processing practices).