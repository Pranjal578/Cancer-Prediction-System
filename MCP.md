# Model Context Protocol (MCP) Status

## Current status

This application does **not** expose or depend on an MCP server. There are no MCP tools, resources, prompts, transports, credentials, or external-service connections in the active runtime.

## Why MCP is absent

The agreed product is a local Streamlit demonstration for a fixed WDBC dataset. Its required flow is local data loading, model comparison, saved-model prediction, and visual reporting. MCP integration would add an external integration surface that is outside the approved scope.

## Boundary

- The Streamlit health route is not an MCP endpoint.
- The saved sklearn model is not an MCP tool.
- The project stores no MCP configuration or secrets.

## Future MCP changes

Do not add MCP support unless the user explicitly requests it. Before implementation, document:

1. The exact tool/resource/prompt contract and input/output schemas.
2. Authentication, authorization, privacy, and retention behavior.
3. Error handling and test coverage.
4. Any impact on `API.md`, `DATABASE.md`, `PRD.md`, `PLAN.md`, `DESIGN.md`, and `TESTING.md`.

Then update this document and the `PLAN.md` change log immediately after every MCP-related change.
