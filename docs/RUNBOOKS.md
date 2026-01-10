# ELEANOR V8 — Operational Runbooks

**Last Updated**: January 8, 2025  
**Version**: 1.0

---

## Table of Contents

1. [Common Operations](#common-operations)
2. [Troubleshooting](#troubleshooting)
3. [Maintenance](#maintenance)
4. [Scaling](#scaling)
5. [Monitoring](#monitoring)

---

## Common Operations

### Health Check

**Purpose**: Verify system health

**Procedure**:
```bash
# Check API health
curl https://api.eleanor-v8.com/health

# Check Kubernetes pods
kubectl get pods -n eleanor-v8

# Check service status
kubectl get svc -n eleanor-v8

# Check deployment status
kubectl rollout status deployment/eleanor-v8 -n eleanor-v8
```

**Expected Output**:
- Health endpoint returns 200 OK
- All pods in Running state
- Services have endpoints
- Deployment shows available replicas

---

### View Logs

**Purpose**: Access application logs

**Procedure**:
```bash
# View recent logs
kubectl logs -n eleanor-v8 -l app=eleanor-v8 --tail=100

# Follow logs
kubectl logs -n eleanor-v8 -l app=eleanor-v8 -f

# View logs for specific pod
kubectl logs -n eleanor-v8 eleanor-v8-<pod-id>

# View logs with grep
kubectl logs -n eleanor-v8 -l app=eleanor-v8 | grep ERROR

# Export logs
kubectl logs -n eleanor-v8 -l app=eleanor-v8 > logs.txt
```

---

### Restart Application

**Purpose**: Restart application pods

**Procedure**:
```bash
# Rolling restart
kubectl rollout restart deployment/eleanor-v8 -n eleanor-v8

# Wait for rollout
kubectl rollout status deployment/eleanor-v8 -n eleanor-v8

# Verify pods restarted
kubectl get pods -n eleanor-v8
```

**When to Use**:
- After configuration changes
- To clear memory leaks
- After dependency updates
- To apply code changes

---

### Scale Application

**Purpose**: Adjust number of replicas

**Procedure**:
```bash
# Scale up
kubectl scale deployment/eleanor-v8 --replicas=5 -n eleanor-v8

# Scale down
kubectl scale deployment/eleanor-v8 --replicas=2 -n eleanor-v8

# Verify scaling
kubectl get pods -n eleanor-v8
kubectl get hpa -n eleanor-v8  # If using HPA
```

**When to Use**:
- Increased traffic
- Performance issues
- Cost optimization
- Maintenance windows

---

## Troubleshooting

### High Error Rate

**Symptoms**:
- Error rate > 1%
- 5xx responses increasing
- Health checks failing

**Diagnosis**:
```bash
# Check error logs
kubectl logs -n eleanor-v8 -l app=eleanor-v8 | grep -i error | tail -50

# Check metrics
curl https://api.eleanor-v8.com/metrics | grep eleanor_requests_total

# Check resource usage
kubectl top pods -n eleanor-v8

# Check dependencies
curl https://api.eleanor-v8.com/health
```

**Common Causes**:
1. Resource exhaustion (CPU/memory)
2. Dependency failures (database, Redis, Weaviate)
3. Configuration errors
4. Code bugs

**Resolution**:
```bash
# If resource exhaustion
kubectl scale deployment/eleanor-v8 --replicas=5 -n eleanor-v8

# If dependency failure
# See dependency-specific runbooks

# If configuration error
kubectl get configmap -n eleanor-v8
kubectl edit configmap eleanor-v8-config -n eleanor-v8
kubectl rollout restart deployment/eleanor-v8 -n eleanor-v8

# If code bug
# Rollback to previous version
kubectl rollout undo deployment/eleanor-v8 -n eleanor-v8
```

---

### High Latency

**Symptoms**:
- P95 latency > 2s
- P99 latency > 5s
- Timeout errors

**Diagnosis**:
```bash
# Check latency metrics
curl https://api.eleanor-v8.com/metrics | grep eleanor_deliberate_duration_seconds

# Check resource usage
kubectl top pods -n eleanor-v8

# Check cache hit rates
curl https://api.eleanor-v8.com/admin/cache/health

# Check database performance
kubectl exec -n eleanor-db postgres-0 -- \
  psql -U eleanor -d eleanor_v8 -c "SELECT * FROM pg_stat_activity WHERE state = 'active'"
```

**Common Causes**:
1. Cache misses
2. Database slow queries
3. LLM API latency
4. Resource constraints

**Resolution**:
```bash
# If cache misses
# Warm cache
curl -X POST https://api.eleanor-v8.com/admin/cache/warm

# If database slow queries
# Check and optimize queries
# Consider read replicas

# If LLM API latency
# Check LLM provider status
# Consider fallback models

# If resource constraints
kubectl scale deployment/eleanor-v8 --replicas=5 -n eleanor-v8
```

---

### Memory Leak

**Symptoms**:
- Memory usage continuously increasing
- Pods being OOM killed
- Performance degradation

**Diagnosis**:
```bash
# Monitor memory usage
kubectl top pods -n eleanor-v8 --watch

# Check for OOM kills
kubectl get events -n eleanor-v8 --field-selector reason=OOMKilling

# Check memory metrics
curl https://api.eleanor-v8.com/metrics | grep process_resident_memory_bytes
```

**Resolution**:
```bash
# Immediate: Restart pods
kubectl rollout restart deployment/eleanor-v8 -n eleanor-v8

# Temporary: Increase memory limits
kubectl set resources deployment/eleanor-v8 \
  --limits=memory=2Gi \
  -n eleanor-v8

# Long-term: Fix memory leak in code
# Review code for:
# - Unclosed connections
# - Growing caches
# - Memory leaks in async code
```

---

### Database Connection Issues

**Symptoms**:
- Database connection errors
- Evidence recording failures
- Precedent retrieval failures

**Diagnosis**:
```bash
# Check database status
kubectl get pods -n eleanor-db

# Test connectivity
kubectl exec -it -n eleanor-v8 eleanor-v8-<pod-id> -- \
  python -c "import asyncpg; asyncio.run(asyncpg.connect('postgresql://...'))"

# Check connection pool
curl https://api.eleanor-v8.com/admin/database/health
```

**Resolution**:
```bash
# If database pod down
kubectl delete pod -n eleanor-db postgres-0
kubectl wait --for=condition=ready pod -n eleanor-db -l app=postgres

# If connection pool exhausted
# Increase pool size in configuration
kubectl edit configmap eleanor-v8-config -n eleanor-v8
# Update DB_POOL_SIZE

# If network issues
# Check network policies
kubectl get networkpolicies -n eleanor-v8
```

---

### Cache Issues

**Symptoms**:
- Low cache hit rates
- Redis connection errors
- Performance degradation

**Diagnosis**:
```bash
# Check Redis status
kubectl get pods -n eleanor-cache

# Check cache health
curl https://api.eleanor-v8.com/admin/cache/health

# Check cache metrics
curl https://api.eleanor-v8.com/metrics | grep cache
```

**Resolution**:
```bash
# If Redis pod down
kubectl delete pod -n eleanor-cache redis-0
kubectl wait --for=condition=ready pod -n eleanor-cache -l app=redis

# If cache hit rate low
# Warm cache
curl -X POST https://api.eleanor-v8.com/admin/cache/warm

# If memory issues
# Increase Redis memory
kubectl set resources deployment/redis \
  --limits=memory=4Gi \
  -n eleanor-cache
```

---

## Maintenance

### Configuration Updates

**Procedure**:
```bash
# Update ConfigMap
kubectl edit configmap eleanor-v8-config -n eleanor-v8

# Or apply from file
kubectl apply -f k8s/configmap.yaml -n eleanor-v8

# Restart to apply changes
kubectl rollout restart deployment/eleanor-v8 -n eleanor-v8

# Verify
kubectl rollout status deployment/eleanor-v8 -n eleanor-v8
```

---

### Secret Rotation

**Procedure**:
```bash
# Update secret in secrets provider
# AWS Secrets Manager
aws secretsmanager update-secret \
  --secret-id eleanor/prod/jwt-secret \
  --secret-string "new-secret-value"

# HashiCorp Vault
vault kv put secret/eleanor/prod/jwt-secret value="new-secret-value"

# Restart application to pick up new secret
kubectl rollout restart deployment/eleanor-v8 -n eleanor-v8

# Verify
kubectl logs -n eleanor-v8 -l app=eleanor-v8 | grep -i "secret.*refresh"
```

---

### Audit Ledger Index (DynamoDB)

**Purpose**: Provide ordered, multi-writer-safe indexing for the S3 Object Lock audit ledger.

**IAM (least privilege)**:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "LedgerDynamoDBAccess",
      "Effect": "Allow",
      "Action": [
        "dynamodb:DescribeTable",
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:Query",
        "dynamodb:UpdateItem"
      ],
      "Resource": "arn:aws:dynamodb:us-east-1:ACCOUNT_ID:table/eleanor-stone-tablet-ledger-index"
    }
  ]
}
```

**Procedure**:
```bash
export ELEANOR_LEDGER_DDB_TABLE=eleanor-stone-tablet-ledger-index
export ELEANOR_LEDGER_DDB_REGION=us-east-1
export ELEANOR_LEDGER_DDB_LEDGER_ID=default

python3 scripts/create_ledger_ddb_table.py --wait --init-meta
```

**Verify**:
```bash
aws dynamodb describe-table --table-name "$ELEANOR_LEDGER_DDB_TABLE" --region "$ELEANOR_LEDGER_DDB_REGION"

aws dynamodb get-item \
  --table-name "$ELEANOR_LEDGER_DDB_TABLE" \
  --region "$ELEANOR_LEDGER_DDB_REGION" \
  --key '{"ledger_id":{"S":"default"},"seq":{"S":"META"}}'
```

**Schema (required)**:
- **PK**: `ledger_id` (string)
- **SK**: `seq` (string)

**Item conventions**:
- `seq="META"` stores `seq_counter` and `last_hash`.
- `seq="SEQ#000000000000000001"` stores per-record index + `s3_key`.

**Pipeline note**:
- Run `python3 scripts/create_ledger_ddb_table.py --wait --init-meta` once per environment.
- Grant the runtime role the IAM policy above for the ledger table.

---

### Database Maintenance

**Procedure**:
```bash
# Vacuum database
kubectl exec -n eleanor-db postgres-0 -- \
  psql -U eleanor -d eleanor_v8 -c "VACUUM ANALYZE;"

# Check table sizes
kubectl exec -n eleanor-db postgres-0 -- \
  psql -U eleanor -d eleanor_v8 -c "
    SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
    FROM pg_tables
    WHERE schemaname = 'public'
    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
  "

# Archive old evidence records (if needed)
kubectl exec -n eleanor-db postgres-0 -- \
  psql -U eleanor -d eleanor_v8 -c "
    DELETE FROM evidence_records
    WHERE timestamp < NOW() - INTERVAL '90 days';
  "
```

---

### Cache Maintenance

**Procedure**:
```bash
# Clear cache (if needed)
kubectl exec -it -n eleanor-cache redis-0 -- redis-cli FLUSHALL

# Check cache size
kubectl exec -it -n eleanor-cache redis-0 -- redis-cli INFO memory

# Monitor cache operations
kubectl exec -it -n eleanor-cache redis-0 -- redis-cli MONITOR
```

---

## Scaling

### Horizontal Scaling

**Procedure**:
```bash
# Manual scaling
kubectl scale deployment/eleanor-v8 --replicas=10 -n eleanor-v8

# Auto-scaling (HPA)
kubectl apply -f k8s/hpa.yaml -n eleanor-v8

# Verify
kubectl get hpa -n eleanor-v8
kubectl get pods -n eleanor-v8
```

**HPA Configuration**:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: eleanor-v8-hpa
  namespace: eleanor-v8
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: eleanor-v8
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

### Vertical Scaling

**Procedure**:
```bash
# Update resource limits
kubectl set resources deployment/eleanor-v8 \
  --requests=cpu=500m,memory=1Gi \
  --limits=cpu=2000m,memory=4Gi \
  -n eleanor-v8

# Verify
kubectl describe deployment/eleanor-v8 -n eleanor-v8 | grep -A 5 Resources
```

---

## Monitoring

### Key Metrics to Monitor

1. **Request Rate**: `eleanor_deliberate_requests_total`
2. **Error Rate**: `eleanor_deliberate_requests_total{outcome="error"}`
3. **Latency**: `eleanor_deliberate_duration_seconds`
4. **Cache Hit Rate**: `cache_hits / (cache_hits + cache_misses)`
5. **Circuit Breaker State**: `circuit_breaker_state`
6. **Resource Usage**: CPU, memory, disk

### Alerting Thresholds

- **Error Rate**: > 1% for 5 minutes
- **Latency P95**: > 2s for 5 minutes
- **Latency P99**: > 5s for 5 minutes
- **CPU Usage**: > 80% for 10 minutes
- **Memory Usage**: > 85% for 10 minutes
- **Cache Hit Rate**: < 50% for 15 minutes
- **Circuit Breaker Open**: Any circuit open for 5 minutes

---

## Emergency Contacts

- **On-Call Engineer**: [Contact]
- **Team Lead**: [Contact]
- **Engineering Manager**: [Contact]
- **Infrastructure Team**: [Contact]

---

**Document Owner**: DevOps Team  
**Review Schedule**: Monthly  
**Last Review**: January 8, 2025
