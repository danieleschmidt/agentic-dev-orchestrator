#!/usr/bin/env python3
"""
Compliance Manager v4.0
Ensures global regulatory compliance (GDPR, CCPA, PIPL, etc.)
"""

import json
import hashlib
import logging
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field

from .i18n_manager import SupportedLocale, RegionalCompliance

logger = logging.getLogger(__name__)


class ComplianceLevel(Enum):
    """Compliance enforcement levels"""
    STRICT = "strict"      # Full enforcement, block non-compliant operations
    MODERATE = "moderate"  # Warn and log non-compliant operations
    RELAXED = "relaxed"    # Log only for audit purposes


class DataClassification(Enum):
    """Data sensitivity classifications"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PII = "pii"  # Personally Identifiable Information
    PHI = "phi"  # Protected Health Information


@dataclass
class ConsentRecord:
    """User consent record for GDPR/privacy compliance"""
    user_id: str
    consent_type: str
    granted: bool
    timestamp: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    withdrawal_timestamp: Optional[datetime] = None
    legal_basis: str = "consent"  # consent, legitimate_interest, contract, etc.
    
    @property
    def is_valid(self) -> bool:
        """Check if consent is still valid"""
        return self.granted and self.withdrawal_timestamp is None
    
    @property
    def age_days(self) -> int:
        """Age of consent in days"""
        return (datetime.now() - self.timestamp).days


@dataclass
class DataProcessingRecord:
    """Record of data processing activity for audit"""
    record_id: str
    activity_type: str
    data_subject: Optional[str]
    data_classification: DataClassification
    processing_purpose: str
    legal_basis: str
    timestamp: datetime
    locale: SupportedLocale
    retention_days: int
    automated_decision: bool = False
    third_party_sharing: bool = False
    cross_border_transfer: bool = False
    
    def to_audit_log(self) -> Dict[str, Any]:
        """Convert to audit log format"""
        return {
            "record_id": self.record_id,
            "activity_type": self.activity_type,
            "data_subject": self.data_subject,
            "data_classification": self.data_classification.value,
            "processing_purpose": self.processing_purpose,
            "legal_basis": self.legal_basis,
            "timestamp": self.timestamp.isoformat(),
            "locale": self.locale.code,
            "retention_days": self.retention_days,
            "automated_decision": self.automated_decision,
            "third_party_sharing": self.third_party_sharing,
            "cross_border_transfer": self.cross_border_transfer
        }


class ComplianceManager:
    """Global regulatory compliance management system"""
    
    def __init__(self, 
                 compliance_level: ComplianceLevel = ComplianceLevel.STRICT,
                 audit_log_path: str = "compliance/audit.log"):
        
        self.compliance_level = compliance_level
        self.audit_log_path = Path(audit_log_path)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Compliance storage
        self.consent_records: Dict[str, List[ConsentRecord]] = {}
        self.processing_records: List[DataProcessingRecord] = []
        
        # Regional compliance cache
        self._compliance_cache: Dict[SupportedLocale, RegionalCompliance] = {}
        
        # Initialize
        self._load_consent_records()
        self._setup_retention_policies()
    
    def _setup_retention_policies(self):
        """Setup data retention policies by region"""
        self.retention_policies = {
            # GDPR regions (EU)
            SupportedLocale.EN_GB: {"default": 2555, "sensitive": 1095},  # 7 years / 3 years
            SupportedLocale.DE_DE: {"default": 2555, "sensitive": 1095},
            SupportedLocale.FR_FR: {"default": 2555, "sensitive": 1095},
            
            # CCPA regions (California)
            SupportedLocale.EN_US: {"default": 2555, "sensitive": 1825},  # 7 years / 5 years
            
            # PIPL regions (China)
            SupportedLocale.ZH_CN: {"default": 1095, "sensitive": 730},   # 3 years / 2 years
            
            # APPI regions (Japan)
            SupportedLocale.JA_JP: {"default": 1825, "sensitive": 1095},  # 5 years / 3 years
        }
    
    def _load_consent_records(self):
        """Load existing consent records"""
        consent_file = self.audit_log_path.parent / "consent_records.json"
        if consent_file.exists():
            try:
                with open(consent_file, 'r') as f:
                    data = json.load(f)
                
                for user_id, records in data.items():
                    self.consent_records[user_id] = []
                    for record_data in records:
                        consent = ConsentRecord(
                            user_id=record_data['user_id'],
                            consent_type=record_data['consent_type'],
                            granted=record_data['granted'],
                            timestamp=datetime.fromisoformat(record_data['timestamp']),
                            ip_address=record_data.get('ip_address'),
                            user_agent=record_data.get('user_agent'),
                            legal_basis=record_data.get('legal_basis', 'consent')
                        )
                        
                        if record_data.get('withdrawal_timestamp'):
                            consent.withdrawal_timestamp = datetime.fromisoformat(
                                record_data['withdrawal_timestamp']
                            )
                        
                        self.consent_records[user_id].append(consent)
                
                logger.info(f"Loaded consent records for {len(self.consent_records)} users")
                
            except Exception as e:
                logger.error(f"Failed to load consent records: {e}")
    
    def _save_consent_records(self):
        """Save consent records to persistent storage"""
        consent_file = self.audit_log_path.parent / "consent_records.json"
        
        try:
            data = {}
            for user_id, records in self.consent_records.items():
                data[user_id] = []
                for consent in records:
                    record_data = {
                        'user_id': consent.user_id,
                        'consent_type': consent.consent_type,
                        'granted': consent.granted,
                        'timestamp': consent.timestamp.isoformat(),
                        'legal_basis': consent.legal_basis
                    }
                    
                    if consent.ip_address:
                        record_data['ip_address'] = consent.ip_address
                    if consent.user_agent:
                        record_data['user_agent'] = consent.user_agent
                    if consent.withdrawal_timestamp:
                        record_data['withdrawal_timestamp'] = consent.withdrawal_timestamp.isoformat()
                    
                    data[user_id].append(record_data)
            
            with open(consent_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save consent records: {e}")
    
    def record_consent(self, 
                      user_id: str,
                      consent_type: str,
                      granted: bool,
                      ip_address: Optional[str] = None,
                      user_agent: Optional[str] = None,
                      legal_basis: str = "consent") -> ConsentRecord:
        """Record user consent for compliance"""
        
        consent_record = ConsentRecord(
            user_id=user_id,
            consent_type=consent_type,
            granted=granted,
            timestamp=datetime.now(),
            ip_address=ip_address,
            user_agent=user_agent,
            legal_basis=legal_basis
        )
        
        if user_id not in self.consent_records:
            self.consent_records[user_id] = []
        
        self.consent_records[user_id].append(consent_record)
        self._save_consent_records()
        
        # Log for audit
        self._log_audit_event({
            "event_type": "consent_recorded",
            "user_id": user_id,
            "consent_type": consent_type,
            "granted": granted,
            "legal_basis": legal_basis,
            "timestamp": consent_record.timestamp.isoformat()
        })
        
        logger.info(f"Consent recorded for user {user_id}: {consent_type} = {granted}")
        return consent_record
    
    def withdraw_consent(self, user_id: str, consent_type: str) -> bool:
        """Withdraw user consent"""
        
        if user_id not in self.consent_records:
            return False
        
        # Find and withdraw the most recent consent of this type
        for consent in reversed(self.consent_records[user_id]):
            if consent.consent_type == consent_type and consent.granted:
                consent.withdrawal_timestamp = datetime.now()
                consent.granted = False
                
                self._save_consent_records()
                
                # Log withdrawal
                self._log_audit_event({
                    "event_type": "consent_withdrawn",
                    "user_id": user_id,
                    "consent_type": consent_type,
                    "withdrawal_timestamp": consent.withdrawal_timestamp.isoformat()
                })
                
                logger.info(f"Consent withdrawn for user {user_id}: {consent_type}")
                return True
        
        return False
    
    def check_consent(self, user_id: str, consent_type: str) -> bool:
        """Check if user has valid consent"""
        
        if user_id not in self.consent_records:
            return False
        
        # Find the most recent consent of this type
        for consent in reversed(self.consent_records[user_id]):
            if consent.consent_type == consent_type:
                return consent.is_valid
        
        return False
    
    def record_data_processing(self,
                             activity_type: str,
                             data_classification: DataClassification,
                             processing_purpose: str,
                             locale: SupportedLocale,
                             data_subject: Optional[str] = None,
                             legal_basis: str = "legitimate_interest",
                             automated_decision: bool = False,
                             third_party_sharing: bool = False) -> DataProcessingRecord:
        """Record data processing activity"""
        
        # Get regional compliance requirements
        regional_compliance = RegionalCompliance.get_compliance_for_locale(locale)
        
        # Check if cross-border transfer
        cross_border = locale.country_code != SupportedLocale.EN_US.country_code
        
        record = DataProcessingRecord(
            record_id=self._generate_record_id(),
            activity_type=activity_type,
            data_subject=data_subject,
            data_classification=data_classification,
            processing_purpose=processing_purpose,
            legal_basis=legal_basis,
            timestamp=datetime.now(),
            locale=locale,
            retention_days=regional_compliance.data_retention_days,
            automated_decision=automated_decision,
            third_party_sharing=third_party_sharing,
            cross_border_transfer=cross_border
        )
        
        self.processing_records.append(record)
        
        # Log for audit
        self._log_audit_event(record.to_audit_log())
        
        # Check compliance requirements
        self._validate_processing_compliance(record, regional_compliance)
        
        logger.debug(f"Data processing recorded: {activity_type} for {locale.code}")
        return record
    
    def _validate_processing_compliance(self, 
                                      record: DataProcessingRecord, 
                                      compliance: RegionalCompliance):
        """Validate processing activity against compliance requirements"""
        
        issues = []
        
        # Check consent requirements
        if (compliance.consent_management and 
            record.data_classification in [DataClassification.PII, DataClassification.PHI] and
            record.data_subject and
            not self.check_consent(record.data_subject, "data_processing")):
            
            issues.append(f"Missing consent for {record.data_classification.value} processing")
        
        # Check data residency requirements
        if compliance.data_residency and record.cross_border_transfer:
            issues.append("Cross-border transfer violates data residency requirements")
        
        # Check encryption requirements
        if compliance.encryption_required and record.data_classification in [
            DataClassification.CONFIDENTIAL, 
            DataClassification.RESTRICTED,
            DataClassification.PII,
            DataClassification.PHI
        ]:
            # This would integrate with actual encryption checks
            pass
        
        # Handle compliance issues
        if issues:
            self._handle_compliance_violation(record, issues, compliance)
    
    def _handle_compliance_violation(self, 
                                   record: DataProcessingRecord,
                                   issues: List[str],
                                   compliance: RegionalCompliance):
        """Handle compliance violations based on enforcement level"""
        
        violation_event = {
            "event_type": "compliance_violation",
            "record_id": record.record_id,
            "locale": record.locale.code,
            "regulation": compliance.privacy_regulation,
            "issues": issues,
            "timestamp": datetime.now().isoformat(),
            "enforcement_level": self.compliance_level.value
        }
        
        self._log_audit_event(violation_event)
        
        if self.compliance_level == ComplianceLevel.STRICT:
            raise ComplianceViolationError(
                f"Compliance violation in {record.locale.code}: {'; '.join(issues)}"
            )
        elif self.compliance_level == ComplianceLevel.MODERATE:
            logger.warning(f"Compliance warning for {record.locale.code}: {'; '.join(issues)}")
        else:  # RELAXED
            logger.info(f"Compliance note for {record.locale.code}: {'; '.join(issues)}")
    
    def _generate_record_id(self) -> str:
        """Generate unique record ID"""
        timestamp = datetime.now().isoformat()
        data = f"{timestamp}_{len(self.processing_records)}"
        return hashlib.md5(data.encode()).hexdigest()[:12]
    
    def _log_audit_event(self, event: Dict[str, Any]):
        """Log event for compliance audit"""
        try:
            with open(self.audit_log_path, 'a') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def cleanup_expired_data(self) -> Dict[str, int]:
        """Clean up data that has exceeded retention periods"""
        
        cleanup_stats = {"records_removed": 0, "consents_expired": 0}
        current_time = datetime.now()
        
        # Clean up processing records
        valid_records = []
        for record in self.processing_records:
            age_days = (current_time - record.timestamp).days
            if age_days <= record.retention_days:
                valid_records.append(record)
            else:
                cleanup_stats["records_removed"] += 1
                
                # Log data deletion
                self._log_audit_event({
                    "event_type": "data_deleted",
                    "record_id": record.record_id,
                    "reason": "retention_period_expired",
                    "age_days": age_days,
                    "retention_days": record.retention_days,
                    "timestamp": current_time.isoformat()
                })
        
        self.processing_records = valid_records
        
        # Clean up expired consents (older than 2 years without renewal)
        for user_id, consents in list(self.consent_records.items()):
            valid_consents = []
            for consent in consents:
                if consent.age_days <= 730:  # 2 years
                    valid_consents.append(consent)
                else:
                    cleanup_stats["consents_expired"] += 1
            
            if valid_consents:
                self.consent_records[user_id] = valid_consents
            else:
                del self.consent_records[user_id]
        
        self._save_consent_records()
        
        logger.info(f"Cleanup completed: {cleanup_stats}")
        return cleanup_stats
    
    def generate_compliance_report(self, 
                                 locale: Optional[SupportedLocale] = None) -> Dict[str, Any]:
        """Generate compliance report for audit purposes"""
        
        current_time = datetime.now()
        
        # Filter records by locale if specified
        records = self.processing_records
        if locale:
            records = [r for r in records if r.locale == locale]
        
        # Consent analysis
        total_users = len(self.consent_records)
        active_consents = sum(
            1 for consents in self.consent_records.values()
            for consent in consents if consent.is_valid
        )
        
        # Processing activity analysis
        activity_types = {}
        data_classifications = {}
        for record in records:
            activity_types[record.activity_type] = activity_types.get(record.activity_type, 0) + 1
            data_classifications[record.data_classification.value] = \
                data_classifications.get(record.data_classification.value, 0) + 1
        
        # Regional compliance status
        regional_status = {}
        if locale:
            compliance = RegionalCompliance.get_compliance_for_locale(locale)
            regional_status = {
                "regulation": compliance.privacy_regulation,
                "data_residency_required": compliance.data_residency,
                "encryption_required": compliance.encryption_required,
                "consent_management_required": compliance.consent_management,
                "retention_days": compliance.data_retention_days
            }
        
        report = {
            "report_timestamp": current_time.isoformat(),
            "scope": locale.code if locale else "global",
            "summary": {
                "total_processing_records": len(records),
                "total_users_with_consent": total_users,
                "active_consents": active_consents,
                "compliance_level": self.compliance_level.value
            },
            "activity_breakdown": activity_types,
            "data_classification_breakdown": data_classifications,
            "regional_compliance": regional_status,
            "retention_status": self._analyze_retention_status(records),
            "recent_violations": self._get_recent_violations()
        }
        
        return report
    
    def _analyze_retention_status(self, records: List[DataProcessingRecord]) -> Dict[str, Any]:
        """Analyze data retention status"""
        current_time = datetime.now()
        
        retention_buckets = {"current": 0, "expiring_soon": 0, "expired": 0}
        
        for record in records:
            age_days = (current_time - record.timestamp).days
            days_until_expiry = record.retention_days - age_days
            
            if days_until_expiry > 30:
                retention_buckets["current"] += 1
            elif days_until_expiry > 0:
                retention_buckets["expiring_soon"] += 1
            else:
                retention_buckets["expired"] += 1
        
        return retention_buckets
    
    def _get_recent_violations(self) -> List[Dict[str, Any]]:
        """Get recent compliance violations from audit log"""
        violations = []
        
        try:
            if self.audit_log_path.exists():
                with open(self.audit_log_path, 'r') as f:
                    for line in f:
                        try:
                            event = json.loads(line.strip())
                            if event.get('event_type') == 'compliance_violation':
                                event_time = datetime.fromisoformat(event['timestamp'])
                                if (datetime.now() - event_time).days <= 7:  # Last 7 days
                                    violations.append(event)
                        except json.JSONDecodeError:
                            continue
        
        except Exception as e:
            logger.error(f"Failed to read audit log for violations: {e}")
        
        return violations[-10:]  # Last 10 violations
    
    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all data for a user (GDPR Article 20 - Right to data portability)"""
        
        user_data = {
            "user_id": user_id,
            "export_timestamp": datetime.now().isoformat(),
            "consent_records": [],
            "processing_records": []
        }
        
        # Export consent records
        if user_id in self.consent_records:
            for consent in self.consent_records[user_id]:
                user_data["consent_records"].append({
                    "consent_type": consent.consent_type,
                    "granted": consent.granted,
                    "timestamp": consent.timestamp.isoformat(),
                    "legal_basis": consent.legal_basis,
                    "withdrawal_timestamp": consent.withdrawal_timestamp.isoformat() 
                                          if consent.withdrawal_timestamp else None
                })
        
        # Export processing records
        for record in self.processing_records:
            if record.data_subject == user_id:
                user_data["processing_records"].append(record.to_audit_log())
        
        # Log data export
        self._log_audit_event({
            "event_type": "data_exported",
            "user_id": user_id,
            "export_timestamp": user_data["export_timestamp"],
            "records_count": len(user_data["processing_records"]),
            "consents_count": len(user_data["consent_records"])
        })
        
        return user_data
    
    def delete_user_data(self, user_id: str, reason: str = "user_request") -> bool:
        """Delete all data for a user (GDPR Article 17 - Right to erasure)"""
        
        deleted_records = 0
        
        # Remove consent records
        if user_id in self.consent_records:
            deleted_records += len(self.consent_records[user_id])
            del self.consent_records[user_id]
        
        # Remove processing records
        original_count = len(self.processing_records)
        self.processing_records = [
            record for record in self.processing_records 
            if record.data_subject != user_id
        ]
        deleted_records += original_count - len(self.processing_records)
        
        self._save_consent_records()
        
        # Log data deletion
        self._log_audit_event({
            "event_type": "user_data_deleted",
            "user_id": user_id,
            "reason": reason,
            "deleted_records_count": deleted_records,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Deleted {deleted_records} records for user {user_id}")
        return deleted_records > 0


class ComplianceViolationError(Exception):
    """Raised when a compliance violation occurs in strict mode"""
    pass


# Global compliance manager instance
compliance_manager = ComplianceManager()


def ensure_compliance(locale: SupportedLocale, 
                     data_classification: DataClassification = DataClassification.INTERNAL):
    """Decorator to ensure compliance for operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Record the data processing activity
            compliance_manager.record_data_processing(
                activity_type=func.__name__,
                data_classification=data_classification,
                processing_purpose=func.__doc__ or "System operation",
                locale=locale
            )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator