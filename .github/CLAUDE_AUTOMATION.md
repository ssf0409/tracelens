# Maintaining Claude Automation

The Claude workflows are separate from TraceLens tests, builds, and releases.
A failed Claude execution is not a code-review finding, and a completed
execution is not approval to merge. Check the review against the current PR head.

## Authentication and Setup

Both workflows use the official Claude GitHub App and the repository secret
`CLAUDE_CODE_OAUTH_TOKEN`. Generate a long-lived subscription token with
`claude setup-token`, then update it through GitHub's Actions secrets settings
or `gh secret set CLAUDE_CODE_OAUTH_TOKEN --repo ssf0409/tracelens` (hidden prompt).
Never paste the token into an issue, PR, workflow file, or command-line argument.
Subscription authentication remains supported; changing to API-key billing is
a separate maintainer decision, not an automatic fallback.

## Failure Diagnosis

1. Check whether setup, GitHub App authentication, or Claude execution failed.
2. Read the **Claude execution diagnostics** job summary. It reports validated
   HTTP status codes and recognized OAuth errors, never raw model/tool output.
3. HTTP 401 indicates rejected authentication; an explicit expired/invalid-token
   hint warrants replacement. HTTP 400 alone does not prove a bad token.
   Check account/model access for 403/404, usage limits for 429, and provider
   health for 5xx. Missing diagnostics require inspecting the action setup logs.
4. Fix the identified cause, rerun the failed job, and verify that Claude actually
   reviewed the intended PR head. Do not turn on `show_full_output` or upload the
   execution transcript to this public repository to debug a failure.

## Workflow Changes

The official GitHub App validates workflow content against the default branch.
On a PR changing a Claude workflow, the action can skip execution and still
return success. Our completion check intentionally fails in that case.
Maintainers must review such a workflow change manually, verify ordinary CI,
and merge it before testing Claude against a subsequent PR. Do not bypass the
validation with a more privileged token or a `pull_request_target` trigger.

The review prompt includes `--comment`, and `claude_args` enables the inline
comment tool. Without those settings, a successful review can remain only in
the action log. The upstream plugin may still skip drafts, trivial changes,
or PRs it considers already reviewed; inspect the actual result.

See the [official setup and review documentation](https://code.claude.com/docs/en/github-actions)
and the [action output contract](https://github.com/anthropics/claude-code-action/blob/v1/action.yml).
