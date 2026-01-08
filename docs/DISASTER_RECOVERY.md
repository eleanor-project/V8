# ELEANOR V8 — Disaster Recovery Plan

**Last Updated**: January 8, 2025  
**Version**: 1.0

---

## Executive Summary

This document outlines the disaster recovery procedures for ELEANOR V8 in production. It covers recovery strategies, procedures, and testing requirements.

---

## Recovery Objectives

### Recovery Time Objectives (RTO)

- **Critical Systems**: 15 minutes
- **Non-Critical Systems**: 1 hour
- **Full Service Restoration**: 4 hours

### Recovery Point Objectives (RPO)

- **Data Loss Tolerance**: 5 minutes
- **Evidence Records**: Zero data loss (immutable audit log)
- **Configuration**: Zero data loss (version controlled)

---

## Disaster Scenarios

### 1. Application Failure

**Symptoms**:
- API endpoints returning 5xx errors
- Health checks failing
- High error rates in monitoring

**Recovery Steps**:

1. **Immediate Response** (0-5 minutes)
   ```bash
   # Check application status
   kubectl get pods -n eleanor-v8
   kubectl logs -n eleanor-v8 -l app=eleanor-v8 --tail=100
   
   # Check health endpoint
   curl https://api.eleanor-v8.com/health
   ```

2. **Diagnosis** (5-10 minutes)
   - Review application logs
   - Check resource utilization (CPU, memory)
   - Review error patterns
   - Check dependency health (database, Redis, Weaviate)

3. **Recovery Actions** (10-15 minutes)
   ```bash
   # Restart application if needed
   kubectl rollout restart deployment/eleanor-v8 -n eleanor-v8
   
   # Scale up if resource constrained
   kubectl scale deployment/eleanor-v8 --replicas=5 -n eleanor-v8
   
   # Check recovery
   kubectl rollout status deployment/eleanor-v8 -n eleanor-v8
   ```

4. **Verification** (15-20 minutes)
   - Verify health checks passing
   - Test critical endpoints
   - Monitor error rates
   - Verify metrics returning to normal

---

### 2. Database Failure

**Symptoms**:
- Database connection errors
- Evidence recording failures
- Precedent retrieval failures

**Recovery Steps**:

1. **Immediate Response** (0-5 minutes)
   ```bash
   # Check database status
   kubectl get pods -n eleanor-db
   kubectl logs -n eleanor-db -l app=postgres --tail=100
   
   # Test database connectivity
   kubectl exec -it -n eleanor-db postgres-0 -- psql -U eleanor -d eleanor_v8 -c "SELECT 1"
   ```

2. **Diagnosis** (5-10 minutes)
   - Check database pod status
   - Review database logs
   - Check disk space
   - Verify backup status

3. **Recovery Actions** (10-30 minutes)
   
   **Option A: Pod Restart** (if pod failure)
   ```bash
   kubectl delete pod -n eleanor-db postgres-0
   # Wait for pod to restart
   kubectl wait --for=condition=ready pod -n eleanor-db -l app=postgres --timeout=5m
   ```
   
   **Option B: Restore from Backup** (if data corruption)
   ```bash
   # Restore from latest backup
   ./scripts/restore_database.sh --backup=latest
   
   # Verify restoration
   kubectl exec -it -n eleanor-db postgres-0 -- psql -U eleanor -d eleanor_v8 -c "SELECT COUNT(*) FROM evidence_records"
   ```

4. **Verification** (30-35 minutes)
   - Verify database connectivity
   - Test evidence recording
   - Verify data integrity
   - Monitor application health

---

### 3. Cache/Redis Failure

**Symptoms**:
- Cache misses increasing
- Redis connection errors
- Performance degradation

**Recovery Steps**:

1. **Immediate Response** (0-5 minutes)
   ```bash
   # Check Redis status
   kubectl get pods -n eleanor-cache
   kubectl logs -n eleanor-cache -l app=redis --tail=100
   ```

2. **Diagnosis** (5-10 minutes)
   - Check Redis pod status
   - Review Redis logs
   - Check memory usage
   - Verify cluster status (if using Redis Cluster)

3. **Recovery Actions** (10-20 minutes)
   
   **Option A: Pod Restart** (if pod failure)
   ```bash
   kubectl delete pod -n eleanor-cache redis-0
   kubectl wait --for=condition=ready pod -n eleanor-cache -l app=redis --timeout=5m
   ```
   
   **Option B: Failover to Standby** (if using Redis Sentinel)
   ```bash
   # Sentinel will automatically failover
   # Verify new master
   kubectl exec -it -n eleanor-cache redis-sentinel-0 -- redis-cli SENTINEL get-master-addr-by-name mymaster
   ```

4. **Verification** (20-25 minutes)
   - Verify Redis connectivity
   - Test cache operations
   - Monitor cache hit rates
   - Verify application performance

---

### 4. Vector Database (Weaviate) Failure

**Symptoms**:
- Precedent retrieval failures
- Embedding storage failures
- Similarity search errors

**Recovery Steps**:

1. **Immediate Response** (0-5 minutes)
   ```bash
   # Check Weaviate status
   kubectl get pods -n eleanor-vector
   kubectl logs -n eleanor-vector -l app=weaviate --tail=100
   ```

2. **Diagnosis** (5-10 minutes)
   - Check Weaviate pod status
   - Review Weaviate logs
   - Check disk space
   - Verify backup status

3. **Recovery Actions** (10-30 minutes)
   
   **Option A: Pod Restart**
   ```bash
   kubectl delete pod -n eleanor-vector weaviate-0
   kubectl wait --for=condition=ready pod -n eleanor-vector -l app=weaviate --timeout=5m
   ```
   
   **Option B: Restore from Backup**
   ```bash
   # Restore Weaviate schema and data
   ./scripts/restore_weaviate.sh --backup=latest
   ```

4. **Verification** (30-35 minutes)
   - Verify Weaviate connectivity
   - Test precedent retrieval
   - Verify embedding storage
   - Monitor application health

---

### 5. Secrets Management Failure

**Symptoms**:
- API key retrieval failures
- Authentication failures
- LLM API call failures

**Recovery Steps**:

1. **Immediate Response** (0-5 minutes)
   ```bash
   # Check secrets provider status
   # AWS Secrets Manager
   aws secretsmanager describe-secret --secret-id eleanor/prod/jwt-secret
   
   # HashiCorp Vault
   vault status
   vault read secret/eleanor/prod/jwt-secret
   ```

2. **Diagnosis** (5-10 minutes)
   - Check secrets provider connectivity
   - Verify IAM/authentication
   - Check secret availability
   - Review application logs

3. **Recovery Actions** (10-20 minutes)
   
   **Option A: Fallback to Environment Variables** (temporary)
   ```bash
   # Set environment variables as fallback
   kubectl set env deployment/eleanor-v8 \
     JWT_SECRET=$(aws secretsmanager get-secret-value --secret-id eleanor/prod/jwt-secret --query SecretString --output text) \
     -n eleanor-v8
   ```
   
   **Option B: Fix Secrets Provider**
   ```bash
   # Fix IAM permissions
   # Restart secrets provider
   # Verify connectivity
   ```

4. **Verification** (20-25 minutes)
   - Verify secret retrieval
   - Test authentication
   - Test LLM API calls
   - Monitor application health

---

### 6. Complete Infrastructure Failure

**Symptoms**:
- All services unavailable
- Complete loss of connectivity
- Infrastructure provider outage

**Recovery Steps**:

1. **Immediate Response** (0-15 minutes)
   - Assess scope of failure
   - Activate disaster recovery site (if available)
   - Notify stakeholders

2. **Recovery Actions** (15 minutes - 4 hours)
   
   **Option A: Failover to DR Site**
   ```bash
   # Activate DR environment
   kubectl config use-context dr-cluster
   kubectl apply -f k8s/dr/
   
   # Verify services
   kubectl get pods -n eleanor-v8
   ```
   
   **Option B: Restore from Backup**
   ```bash
   # Restore infrastructure
   terraform apply -var-file=dr.tfvars
   
   # Restore data
   ./scripts/restore_all.sh --backup=latest
   
   # Deploy application
   kubectl apply -f k8s/production/
   ```

3. **Verification** (4-5 hours)
   - Verify all services operational
   - Test critical endpoints
   - Verify data integrity
   - Monitor system health

---

## Backup Procedures

### Database Backups

**Schedule**: Daily at 2 AM UTC

**Retention**: 30 days

**Procedure**:
```bash
#!/bin/bash
# scripts/backup_database.sh

BACKUP_NAME="eleanor-v8-$(date +%Y%m%d-%H%M%S)"
kubectl exec -n eleanor-db postgres-0 -- \
  pg_dump -U eleanor eleanor_v8 | \
  gzip > "backups/db/${BACKUP_NAME}.sql.gz"

# Upload to S3
aws s3 cp "backups/db/${BACKUP_NAME}.sql.gz" \
  "s3://eleanor-backups/database/${BACKUP_NAME}.sql.gz"
```

### Weaviate Backups

**Schedule**: Daily at 3 AM UTC

**Retention**: 30 days

**Procedure**:
```bash
#!/bin/bash
# scripts/backup_weaviate.sh

BACKUP_NAME="weaviate-$(date +%Y%m%d-%H%M%S)"
kubectl exec -n eleanor-vector weaviate-0 -- \
  curl -X POST http://localhost:8080/v1/backups/filesystem \
  -H "Content-Type: application/json" \
  -d "{\"id\": \"${BACKUP_NAME}\", \"include\": \"*\"}"

# Export backup
kubectl cp eleanor-vector/weaviate-0:/var/lib/weaviate/backups/${BACKUP_NAME} \
  "backups/weaviate/${BACKUP_NAME}"

# Upload to S3
aws s3 sync "backups/weaviate/${BACKUP_NAME}" \
  "s3://eleanor-backups/weaviate/${BACKUP_NAME}/"
```

### Configuration Backups

**Schedule**: On every configuration change

**Retention**: Indefinite (version controlled)

**Procedure**:
- All configuration in Git repository
- Automatic backup on commit
- Tagged releases for major changes

---

## Testing Procedures

### Monthly DR Drill

**Schedule**: First Saturday of each month

**Procedure**:
1. Simulate failure scenario
2. Execute recovery procedures
3. Measure recovery time
4. Document lessons learned
5. Update procedures as needed

### Quarterly Full DR Test

**Schedule**: Quarterly

**Procedure**:
1. Failover to DR site
2. Verify all services
3. Test data integrity
4. Failback to primary
5. Document results

---

## Communication Plan

### Incident Notification

1. **Immediate** (0-5 minutes)
   - Alert on-call engineer
   - Notify team lead

2. **Escalation** (5-15 minutes)
   - Notify engineering manager
   - Update status page

3. **Stakeholder Notification** (15-30 minutes)
   - Notify product team
   - Update customers (if service impact)

### Status Updates

- **Every 15 minutes** during incident
- **Post-incident report** within 24 hours
- **Root cause analysis** within 1 week

---

## Recovery Verification Checklist

After any recovery procedure:

- [ ] All health checks passing
- [ ] Critical endpoints responding
- [ ] Error rates normal
- [ ] Data integrity verified
- [ ] Performance metrics normal
- [ ] Monitoring operational
- [ ] Logs accessible
- [ ] Backups current
- [ ] Documentation updated

---

## Post-Incident Procedures

1. **Immediate** (0-24 hours)
   - Document incident timeline
   - Collect logs and metrics
   - Initial root cause analysis

2. **Short-term** (1-7 days)
   - Complete root cause analysis
   - Update procedures
   - Implement fixes
   - Review monitoring/alerting

3. **Long-term** (1-4 weeks)
   - Post-mortem meeting
   - Process improvements
   - Training updates
   - Documentation updates

---

## Contacts

### On-Call Rotation

- **Primary**: [Contact Info]
- **Secondary**: [Contact Info]
- **Escalation**: [Contact Info]

### Vendor Contacts

- **Infrastructure Provider**: [Contact Info]
- **Database Provider**: [Contact Info]
- **Monitoring Provider**: [Contact Info]

---

**Document Owner**: DevOps Team  
**Review Schedule**: Quarterly  
**Last Review**: January 8, 2025
