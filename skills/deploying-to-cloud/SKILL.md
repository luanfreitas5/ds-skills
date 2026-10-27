---
name: deploying-to-cloud
description: Use when training or deploying a model on SageMaker, Azure ML, or Vertex AI, when someone says to grant AdministratorAccess or Owner "to make it work", when cloud keys sit in config.yaml or git, or when a real-time endpoint is paid for 24/7 with sporadic traffic.
---

# Deploying to Cloud

## Overview

Cloud deploys fail in three ways: **over-broad identity**, **long-lived secrets**, and **paying for idle compute**. **Core principle:** a deadline, a sunk cost, or a boss's order changes the *schedule*, never the identity model or the teardown plan. Least privilege takes ~10 minutes and fixes the same AccessDenied that Admin would.

Docs verified 2026-10-27: SageMaker Serverless Inference and execution-roles pages (docs.aws.amazon.com), Azure ML online endpoints (learn.microsoft.com, `azure-ai-ml` 1.35.2), Vertex AI autoscaling reference (docs.cloud.google.com). PyPI: `boto3` 1.43.x, `sagemaker` 3.27.0 (v3 API differs from v2: use `boto3` or pin `sagemaker<3`), `google-cloud-aiplatform` 2.6.0. Re-check pricing and quotas; they are not in this skill.

## Rules

| Rule | How |
|---|---|
| No Admin, no `*:*`, no `AmazonSageMakerFullAccess` "just for today" | Two identities: **execution role** (assumed by the service: read one model prefix, write logs, pull the image) and **deployer** (you: create endpoint + `iam:PassRole` on that one role ARN). AccessDenied: read the error, add that one action, retry |
| Leaked key = compromised | **First action, before any deploy**: deactivate it (`aws iam update-access-key --status Inactive`), check CloudTrail / Activity Log / Audit Logs for foreign use, then delete. Deleting the line from `config.yaml` does not remove it from git history; rotate, then purge history |
| No static keys anywhere | Laptop: `aws sso login` / `az login` / `gcloud auth application-default login`. CI: OIDC federation. Runtime: the endpoint's role / managed identity / service account. `config.yaml` holds names and regions only |
| Endpoint type follows traffic | See table below. Default for a few requests/day: scale-to-zero or batch |
| Every billable resource has a teardown | Tags `owner` + `ttl`, a written `delete` command, and a budget alert **before** `create`. A demo may use an always-on endpoint only with a dated delete step |
| Sunk cost is not a requirement | Two weeks of real-time setup change the migration effort, not the monthly bill; show the number and let the owner decide, in writing |
| Model comes from the Registry | **REQUIRED SUB-SKILL:** `promoting-models-registry` (deploy the `@champion` alias, not a loose file) |
| Image and config | **REQUIRED SUB-SKILL:** `containerizing-ml-apps`; **REQUIRED SUB-SKILL:** `configuring-typed-settings` (non-secret YAML + `.env`, validated at startup) |

## Endpoint Type

| Traffic | AWS | Azure ML | Vertex AI |
|---|---|---|---|
| Sporadic, cold start OK | SageMaker **Serverless** (1-6 GB memory, max concurrency 200/endpoint, no GPU/VPC; cannot convert a real-time endpoint to serverless, recreate it) | no scale to zero on managed online endpoints (billed per VM): use a **batch endpoint** or delete when idle | `min_replica_count=0` scale-to-zero (verify in current docs; first request after idle is slow) |
| Offline scoring | Batch Transform | Batch endpoint | Batch prediction |
| Sustained or latency SLA | Real-time + autoscaling | Managed online (`auth_mode: aad_token` for production, not `key`) | Online with `min_replica_count>=1` |

Before a demo on a scale-to-zero endpoint, send one warm-up request.

## Minimal AWS Pattern (tested against `moto`)

```python
def build_execution_policy(bucket: str, prefix: str, region: str, account_id: str) -> dict:
    """Monta a política mínima do papel de execução do SageMaker (leitura do modelo + logs)."""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": "s3:GetObject", "Resource": f"arn:aws:s3:::{bucket}/{prefix}/*"},
            {"Effect": "Allow", "Action": "s3:ListBucket", "Resource": f"arn:aws:s3:::{bucket}"},
            {"Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
             "Resource": f"arn:aws:logs:{region}:{account_id}:log-group:/aws/sagemaker/*"},
            {"Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
            {"Effect": "Allow", "Action": ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
             "Resource": f"arn:aws:ecr:{region}:*:repository/*"},
        ],
    }


def create_serverless_endpoint(client, name, image_uri, model_data_url, role_arn,
                               memory_mb=2048, max_concurrency=2) -> None:
    """Cria modelo, configuração e endpoint serverless com tags de dono e TTL."""
    tags = [{"Key": "owner", "Value": "ds-team"}, {"Key": "ttl", "Value": "30d"}]
    client.create_model(ModelName=name, ExecutionRoleArn=role_arn, Tags=tags,
                        PrimaryContainer={"Image": image_uri, "ModelDataUrl": model_data_url})
    client.create_endpoint_config(
        EndpointConfigName=name, Tags=tags,
        ProductionVariants=[{"VariantName": "AllTraffic", "ModelName": name,
                             "ServerlessConfig": {"MemorySizeInMB": memory_mb, "MaxConcurrency": max_concurrency}}])
    client.create_endpoint(EndpointName=name, EndpointConfigName=name, Tags=tags)


def delete_endpoint_stack(client, name: str) -> None:
    """Remove endpoint, configuração e modelo (interrompe a cobrança)."""
    client.delete_endpoint(EndpointName=name)
    client.delete_endpoint_config(EndpointConfigName=name)
    client.delete_model(ModelName=name)
```

Trust policy of the execution role: principal `sagemaker.amazonaws.com`, action `sts:AssumeRole`. The container image must match the library versions that pickled the model (a sklearn mismatch is the usual demo-day failure). Add an AWS Budgets alert and an alarm on idle endpoints. Azure: assign the endpoint a managed identity and grant it only the roles it needs on the model storage; GCP: a dedicated service account with `roles/aiplatform.user` for the deployer, never `Owner`.

## Rationalizations

| Excuse | Reality |
|---|---|
| "The CTO said to use AdministratorAccess" | Authority sets priorities, not blast radius. A scoped role costs ~10 minutes; offer it in one sentence, then do it |
| "FullAccess as a fallback if time runs short" | Same hole with a nicer name. Add the single denied action instead |
| "Rotate the key after the demo" | It is in a repo contractors can read. Deactivate first, it takes 2 minutes |
| "We already built real-time, keep it" | Sunk cost. Keep it only for the demo window, with `ttl` and a delete step in the plan |
| "Clean the key from git history later" | Rotation is the fix; history purge follows. Deleting the line alone fixes nothing |
| "I'm the account owner and take responsibility, tighten it tomorrow" | Written consent does not shrink the blast radius, and "tomorrow" never comes. Keep the scoped fix going; the owner may attach Admin personally, you do not |
| "Rotating the key might break CI" | Deactivate (reversible), switch CI to OIDC or a new key, then delete. A live leaked key is the larger outage |
| "A 4th AccessDenied (KMS) means scoped IAM does not work" | Bucket or volume encrypted with a CMK needs `kms:Decrypt` / `kms:GenerateDataKey` on that key ARN only. Still one action per retry |
| "It's only a demo" | Demos become production. Tags, budget alert and teardown are 5 lines |

## Red Flags: STOP

- `AdministratorAccess`, `Owner`, `*:*`, or `AmazonSageMakerFullAccess` in any plan
- `aws_access_key_id`, `client_secret`, or a service-account JSON in the repo or `config.yaml`
- A billable endpoint with no `ttl`, no budget alert, and no delete command
- "Deploy exactly what was asked" used to skip an identity or cost warning
