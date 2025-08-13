#!/usr/bin/env python3
"""
Internationalization Manager v4.0
Comprehensive multi-language support with regional compliance features
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SupportedLocale(Enum):
    """Supported locales with regional information"""
    EN_US = ("en-US", "English (United States)", "Americas/New_York")
    EN_GB = ("en-GB", "English (United Kingdom)", "Europe/London")
    ES_ES = ("es-ES", "Español (España)", "Europe/Madrid")
    ES_MX = ("es-MX", "Español (México)", "America/Mexico_City")
    FR_FR = ("fr-FR", "Français (France)", "Europe/Paris")
    FR_CA = ("fr-CA", "Français (Canada)", "America/Toronto")
    DE_DE = ("de-DE", "Deutsch (Deutschland)", "Europe/Berlin")
    DE_AT = ("de-AT", "Deutsch (Österreich)", "Europe/Vienna")
    IT_IT = ("it-IT", "Italiano (Italia)", "Europe/Rome")
    PT_BR = ("pt-BR", "Português (Brasil)", "America/Sao_Paulo")
    PT_PT = ("pt-PT", "Português (Portugal)", "Europe/Lisbon")
    JA_JP = ("ja-JP", "日本語 (日本)", "Asia/Tokyo")
    KO_KR = ("ko-KR", "한국어 (대한민국)", "Asia/Seoul")
    ZH_CN = ("zh-CN", "中文 (中国)", "Asia/Shanghai")
    ZH_TW = ("zh-TW", "中文 (台灣)", "Asia/Taipei")
    RU_RU = ("ru-RU", "Русский (Россия)", "Europe/Moscow")
    AR_SA = ("ar-SA", "العربية (السعودية)", "Asia/Riyadh")
    HI_IN = ("hi-IN", "हिन्दी (भारत)", "Asia/Kolkata")
    
    @property
    def code(self) -> str:
        return self.value[0]
    
    @property
    def display_name(self) -> str:
        return self.value[1]
    
    @property
    def timezone(self) -> str:
        return self.value[2]
    
    @property
    def language_code(self) -> str:
        return self.code.split('-')[0]
    
    @property
    def country_code(self) -> str:
        return self.code.split('-')[1]


@dataclass
class RegionalCompliance:
    """Regional compliance requirements"""
    locale: SupportedLocale
    privacy_regulation: str  # GDPR, CCPA, LGPD, etc.
    data_residency: bool
    encryption_required: bool
    audit_logging: bool
    consent_management: bool
    data_retention_days: int
    currency_code: str
    date_format: str
    number_format: str
    
    @classmethod
    def get_compliance_for_locale(cls, locale: SupportedLocale) -> 'RegionalCompliance':
        """Get compliance requirements for a locale"""
        compliance_map = {
            SupportedLocale.EN_US: cls(
                locale=locale,
                privacy_regulation="CCPA",
                data_residency=False,
                encryption_required=True,
                audit_logging=True,
                consent_management=True,
                data_retention_days=2555,  # 7 years
                currency_code="USD",
                date_format="%m/%d/%Y",
                number_format="1,234.56"
            ),
            SupportedLocale.EN_GB: cls(
                locale=locale,
                privacy_regulation="GDPR",
                data_residency=True,
                encryption_required=True,
                audit_logging=True,
                consent_management=True,
                data_retention_days=2555,
                currency_code="GBP",
                date_format="%d/%m/%Y",
                number_format="1,234.56"
            ),
            SupportedLocale.DE_DE: cls(
                locale=locale,
                privacy_regulation="GDPR",
                data_residency=True,
                encryption_required=True,
                audit_logging=True,
                consent_management=True,
                data_retention_days=2555,
                currency_code="EUR",
                date_format="%d.%m.%Y",
                number_format="1.234,56"
            ),
            SupportedLocale.FR_FR: cls(
                locale=locale,
                privacy_regulation="GDPR",
                data_residency=True,
                encryption_required=True,
                audit_logging=True,
                consent_management=True,
                data_retention_days=2555,
                currency_code="EUR",
                date_format="%d/%m/%Y",
                number_format="1 234,56"
            ),
            SupportedLocale.JA_JP: cls(
                locale=locale,
                privacy_regulation="APPI",
                data_residency=True,
                encryption_required=True,
                audit_logging=True,
                consent_management=False,
                data_retention_days=1825,  # 5 years
                currency_code="JPY",
                date_format="%Y/%m/%d",
                number_format="1,234"
            ),
            SupportedLocale.ZH_CN: cls(
                locale=locale,
                privacy_regulation="PIPL",
                data_residency=True,
                encryption_required=True,
                audit_logging=True,
                consent_management=True,
                data_retention_days=1095,  # 3 years
                currency_code="CNY",
                date_format="%Y-%m-%d",
                number_format="1,234.56"
            )
        }
        
        # Default compliance for unmapped locales
        return compliance_map.get(locale, cls(
            locale=locale,
            privacy_regulation="GENERIC",
            data_residency=False,
            encryption_required=True,
            audit_logging=True,
            consent_management=True,
            data_retention_days=2555,
            currency_code="USD",
            date_format="%Y-%m-%d",
            number_format="1,234.56"
        ))


@dataclass
class TranslationEntry:
    """Translation entry with metadata"""
    key: str
    value: str
    locale: SupportedLocale
    context: Optional[str] = None
    pluralization_forms: Optional[Dict[str, str]] = None
    variables: Optional[List[str]] = None
    last_updated: Optional[datetime] = None
    translator_notes: Optional[str] = None


class I18nManager:
    """Comprehensive internationalization manager"""
    
    def __init__(self, 
                 default_locale: SupportedLocale = SupportedLocale.EN_US,
                 translations_dir: str = "locales"):
        
        self.default_locale = default_locale
        self.current_locale = default_locale
        self.translations_dir = Path(translations_dir)
        
        # Translation storage
        self.translations: Dict[SupportedLocale, Dict[str, TranslationEntry]] = {}
        self.fallback_chain: List[SupportedLocale] = [
            SupportedLocale.EN_US,  # Ultimate fallback
            SupportedLocale.EN_GB   # Secondary fallback
        ]
        
        # Regional compliance
        self.compliance_requirements = {
            locale: RegionalCompliance.get_compliance_for_locale(locale)
            for locale in SupportedLocale
        }
        
        # Initialize
        self._load_translations()
        self._setup_formatting_rules()
    
    def _setup_formatting_rules(self):
        """Setup locale-specific formatting rules"""
        self.formatting_rules = {
            SupportedLocale.EN_US: {
                "decimal_separator": ".",
                "thousands_separator": ",",
                "currency_symbol": "$",
                "currency_position": "before",
                "date_order": "MDY",
                "time_format": "12h"
            },
            SupportedLocale.DE_DE: {
                "decimal_separator": ",",
                "thousands_separator": ".",
                "currency_symbol": "€",
                "currency_position": "after",
                "date_order": "DMY",
                "time_format": "24h"
            },
            SupportedLocale.FR_FR: {
                "decimal_separator": ",",
                "thousands_separator": " ",
                "currency_symbol": "€",
                "currency_position": "after",
                "date_order": "DMY",
                "time_format": "24h"
            },
            SupportedLocale.JA_JP: {
                "decimal_separator": ".",
                "thousands_separator": ",",
                "currency_symbol": "¥",
                "currency_position": "before",
                "date_order": "YMD",
                "time_format": "24h"
            }
        }
    
    def _load_translations(self):
        """Load all available translations"""
        if not self.translations_dir.exists():
            self.translations_dir.mkdir(parents=True, exist_ok=True)
            self._create_default_translations()
        
        for locale in SupportedLocale:
            locale_file = self.translations_dir / f"{locale.code}.json"
            if locale_file.exists():
                try:
                    with open(locale_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    self.translations[locale] = {}
                    for key, value in data.items():
                        if isinstance(value, dict):
                            # Rich translation entry
                            self.translations[locale][key] = TranslationEntry(
                                key=key,
                                value=value.get('value', str(value)),
                                locale=locale,
                                context=value.get('context'),
                                pluralization_forms=value.get('pluralization_forms'),
                                variables=value.get('variables'),
                                translator_notes=value.get('translator_notes')
                            )
                        else:
                            # Simple string translation
                            self.translations[locale][key] = TranslationEntry(
                                key=key,
                                value=str(value),
                                locale=locale
                            )
                
                except Exception as e:
                    logger.warning(f"Failed to load translations for {locale.code}: {e}")
    
    def _create_default_translations(self):
        """Create default translation files"""
        default_translations = {
            # CLI Messages
            "cli.welcome": "Welcome to Terragon ADO - Autonomous Development Orchestrator v4.0",
            "cli.starting_execution": "Starting autonomous backlog execution with progressive enhancement",
            "cli.environment_check": "Validating ADO Environment...",
            "cli.missing_env_vars": "Missing required environment variables: {vars}",
            "cli.initialization_complete": "ADO initialization complete!",
            "cli.execution_summary": "Execution Summary:",
            "cli.completed_items": "Completed items: {count}",
            "cli.blocked_items": "Blocked items: {count}",
            "cli.escalated_items": "Escalated items: {count}",
            
            # Status Messages
            "status.optimal": "Optimal",
            "status.moderate": "Moderate Load",
            "status.high": "High Load", 
            "status.critical": "Critical",
            "status.healthy": "Healthy",
            "status.degraded": "Degraded",
            "status.unavailable": "Unavailable",
            
            # Error Messages
            "error.connection_failed": "Connection failed",
            "error.timeout": "Operation timeout",
            "error.invalid_input": "Invalid input provided",
            "error.access_denied": "Access denied",
            "error.file_not_found": "File not found",
            "error.processing_failed": "Processing failed",
            
            # Success Messages
            "success.operation_completed": "Operation completed successfully",
            "success.file_saved": "File saved successfully",
            "success.validation_passed": "Validation passed",
            "success.cache_hit": "Cache hit",
            "success.recovery_successful": "Recovery successful",
            
            # Help Text
            "help.commands": "Commands:",
            "help.init_desc": "Initialize ADO in current directory", 
            "help.run_desc": "Execute autonomous backlog processing",
            "help.status_desc": "Show current backlog status",
            "help.validate_desc": "Validate environment and configuration",
            "help.metrics_desc": "Show performance and execution metrics",
            
            # Security Messages
            "security.dangerous_pattern": "Dangerous pattern detected",
            "security.input_rejected": "Input rejected for security reasons",
            "security.validation_failed": "Security validation failed",
            "security.access_logged": "Access logged for audit",
            
            # Performance Messages
            "performance.cache_eviction": "Cache eviction triggered",
            "performance.scaling_up": "Scaling up resources",
            "performance.scaling_down": "Scaling down resources",
            "performance.optimization_complete": "Optimization complete",
            
            # Compliance Messages
            "compliance.gdpr_notice": "This operation is subject to GDPR compliance requirements",
            "compliance.data_retention": "Data retention policy: {days} days",
            "compliance.consent_required": "User consent required for this operation",
            "compliance.audit_logged": "Operation logged for compliance audit",
            
            # Time and Date
            "time.now": "Now",
            "time.today": "Today",
            "time.yesterday": "Yesterday",
            "time.minutes_ago": "{minutes} minutes ago",
            "time.hours_ago": "{hours} hours ago",
            "time.days_ago": "{days} days ago"
        }
        
        # Create translation files for each locale
        locale_translations = {
            SupportedLocale.ES_ES: {
                "cli.welcome": "Bienvenido a Terragon ADO - Orquestador de Desarrollo Autónomo v4.0",
                "cli.starting_execution": "Iniciando ejecución autónoma de backlog con mejora progresiva",
                "cli.environment_check": "Validando entorno ADO...",
                "cli.missing_env_vars": "Variables de entorno requeridas faltantes: {vars}",
                "cli.initialization_complete": "¡Inicialización de ADO completa!",
                "status.optimal": "Óptimo",
                "status.healthy": "Saludable",
                "error.connection_failed": "Conexión falló",
                "success.operation_completed": "Operación completada exitosamente"
            },
            SupportedLocale.FR_FR: {
                "cli.welcome": "Bienvenue dans Terragon ADO - Orchestrateur de Développement Autonome v4.0",
                "cli.starting_execution": "Démarrage de l'exécution autonome du backlog avec amélioration progressive",
                "cli.environment_check": "Validation de l'environnement ADO...",
                "cli.missing_env_vars": "Variables d'environnement requises manquantes : {vars}",
                "cli.initialization_complete": "Initialisation ADO terminée !",
                "status.optimal": "Optimal",
                "status.healthy": "En bonne santé",
                "error.connection_failed": "Échec de la connexion",
                "success.operation_completed": "Opération terminée avec succès"
            },
            SupportedLocale.DE_DE: {
                "cli.welcome": "Willkommen bei Terragon ADO - Autonomer Entwicklungs-Orchestrator v4.0",
                "cli.starting_execution": "Starte autonome Backlog-Ausführung mit progressiver Verbesserung",
                "cli.environment_check": "ADO-Umgebung wird validiert...",
                "cli.missing_env_vars": "Fehlende erforderliche Umgebungsvariablen: {vars}",
                "cli.initialization_complete": "ADO-Initialisierung abgeschlossen!",
                "status.optimal": "Optimal",
                "status.healthy": "Gesund",
                "error.connection_failed": "Verbindung fehlgeschlagen",
                "success.operation_completed": "Operation erfolgreich abgeschlossen"
            },
            SupportedLocale.JA_JP: {
                "cli.welcome": "Terragon ADO - 自律開発オーケストレーター v4.0 へようこそ",
                "cli.starting_execution": "プログレッシブ拡張による自律バックログ実行を開始",
                "cli.environment_check": "ADO環境を検証中...",
                "cli.missing_env_vars": "必要な環境変数が不足しています: {vars}",
                "cli.initialization_complete": "ADOの初期化が完了しました！",
                "status.optimal": "最適",
                "status.healthy": "健全",
                "error.connection_failed": "接続に失敗しました",
                "success.operation_completed": "操作が正常に完了しました"
            },
            SupportedLocale.ZH_CN: {
                "cli.welcome": "欢迎使用 Terragon ADO - 自主开发编排器 v4.0",
                "cli.starting_execution": "开始渐进增强的自主待办事项执行",
                "cli.environment_check": "正在验证 ADO 环境...",
                "cli.missing_env_vars": "缺少必需的环境变量: {vars}",
                "cli.initialization_complete": "ADO 初始化完成！",
                "status.optimal": "最佳",
                "status.healthy": "健康",
                "error.connection_failed": "连接失败",
                "success.operation_completed": "操作成功完成"
            }
        }
        
        # Save English (default) translations
        en_file = self.translations_dir / f"{SupportedLocale.EN_US.code}.json"
        with open(en_file, 'w', encoding='utf-8') as f:
            json.dump(default_translations, f, ensure_ascii=False, indent=2)
        
        # Save translated versions
        for locale, translations in locale_translations.items():
            # Merge with default (English fallback)
            merged_translations = {**default_translations, **translations}
            
            locale_file = self.translations_dir / f"{locale.code}.json"
            with open(locale_file, 'w', encoding='utf-8') as f:
                json.dump(merged_translations, f, ensure_ascii=False, indent=2)
    
    def set_locale(self, locale: Union[SupportedLocale, str]):
        """Set current locale"""
        if isinstance(locale, str):
            # Try to match locale string to SupportedLocale
            for supported_locale in SupportedLocale:
                if (supported_locale.code.lower() == locale.lower() or
                    supported_locale.language_code.lower() == locale.lower()):
                    locale = supported_locale
                    break
            else:
                logger.warning(f"Unsupported locale: {locale}, using default")
                locale = self.default_locale
        
        self.current_locale = locale
        logger.info(f"Locale set to: {locale.display_name} ({locale.code})")
    
    def translate(self, 
                  key: str, 
                  variables: Optional[Dict[str, Any]] = None,
                  locale: Optional[SupportedLocale] = None,
                  count: Optional[int] = None) -> str:
        """Translate a key with variable substitution and pluralization"""
        
        target_locale = locale or self.current_locale
        
        # Try to find translation in target locale
        translation = self._find_translation(key, target_locale)
        
        if not translation:
            # Try fallback locales
            for fallback_locale in self.fallback_chain:
                translation = self._find_translation(key, fallback_locale)
                if translation:
                    break
        
        if not translation:
            # Return key as fallback
            logger.warning(f"No translation found for key: {key}")
            return key
        
        # Handle pluralization
        if count is not None and translation.pluralization_forms:
            plural_key = self._get_plural_form_key(count, target_locale)
            if plural_key in translation.pluralization_forms:
                text = translation.pluralization_forms[plural_key]
            else:
                text = translation.value
        else:
            text = translation.value
        
        # Variable substitution
        if variables:
            try:
                text = text.format(**variables)
            except KeyError as e:
                logger.warning(f"Missing variable in translation {key}: {e}")
        
        return text
    
    def _find_translation(self, key: str, locale: SupportedLocale) -> Optional[TranslationEntry]:
        """Find translation for key in specific locale"""
        return self.translations.get(locale, {}).get(key)
    
    def _get_plural_form_key(self, count: int, locale: SupportedLocale) -> str:
        """Get plural form key based on count and locale rules"""
        # Simplified pluralization rules
        if locale.language_code in ['en']:
            return 'one' if count == 1 else 'other'
        elif locale.language_code in ['es', 'fr', 'de', 'it', 'pt']:
            return 'one' if count == 1 else 'other'
        elif locale.language_code in ['ja', 'ko', 'zh']:
            return 'other'  # No plural forms
        elif locale.language_code in ['ru']:
            # Simplified Russian pluralization
            if count % 10 == 1 and count % 100 != 11:
                return 'one'
            elif 2 <= count % 10 <= 4 and not (12 <= count % 100 <= 14):
                return 'few'
            else:
                return 'many'
        else:
            return 'other'
    
    def format_number(self, 
                     number: Union[int, float], 
                     locale: Optional[SupportedLocale] = None) -> str:
        """Format number according to locale conventions"""
        target_locale = locale or self.current_locale
        rules = self.formatting_rules.get(target_locale, self.formatting_rules[SupportedLocale.EN_US])
        
        # Convert to string with appropriate decimal places
        if isinstance(number, int):
            num_str = str(number)
            decimal_part = ""
        else:
            parts = f"{number:.2f}".split('.')
            num_str = parts[0]
            decimal_part = parts[1] if len(parts) > 1 else ""
        
        # Add thousands separators
        if len(num_str) > 3:
            # Group digits by thousands
            groups = []
            for i in range(len(num_str), 0, -3):
                start = max(0, i - 3)
                groups.append(num_str[start:i])
            num_str = rules["thousands_separator"].join(reversed(groups))
        
        # Add decimal part
        if decimal_part:
            num_str += rules["decimal_separator"] + decimal_part
        
        return num_str
    
    def format_currency(self, 
                       amount: float, 
                       locale: Optional[SupportedLocale] = None) -> str:
        """Format currency according to locale conventions"""
        target_locale = locale or self.current_locale
        rules = self.formatting_rules.get(target_locale, self.formatting_rules[SupportedLocale.EN_US])
        compliance = self.compliance_requirements[target_locale]
        
        formatted_number = self.format_number(amount, target_locale)
        currency_symbol = rules["currency_symbol"]
        
        if rules["currency_position"] == "before":
            return f"{currency_symbol}{formatted_number}"
        else:
            return f"{formatted_number} {currency_symbol}"
    
    def format_date(self, 
                   date: datetime, 
                   locale: Optional[SupportedLocale] = None,
                   include_time: bool = False) -> str:
        """Format date according to locale conventions"""
        target_locale = locale or self.current_locale
        compliance = self.compliance_requirements[target_locale]
        
        date_format = compliance.date_format
        if include_time:
            rules = self.formatting_rules.get(target_locale, self.formatting_rules[SupportedLocale.EN_US])
            if rules["time_format"] == "12h":
                date_format += " %I:%M %p"
            else:
                date_format += " %H:%M"
        
        try:
            return date.strftime(date_format)
        except Exception:
            # Fallback to ISO format
            return date.isoformat()
    
    def get_compliance_requirements(self, locale: Optional[SupportedLocale] = None) -> RegionalCompliance:
        """Get compliance requirements for locale"""
        target_locale = locale or self.current_locale
        return self.compliance_requirements[target_locale]
    
    def is_rtl_locale(self, locale: Optional[SupportedLocale] = None) -> bool:
        """Check if locale uses right-to-left text direction"""
        target_locale = locale or self.current_locale
        rtl_languages = ['ar', 'he', 'fa', 'ur']
        return target_locale.language_code in rtl_languages
    
    def get_supported_locales(self) -> List[Dict[str, str]]:
        """Get list of supported locales with display information"""
        return [
            {
                "code": locale.code,
                "display_name": locale.display_name,
                "language": locale.language_code,
                "country": locale.country_code,
                "timezone": locale.timezone
            }
            for locale in SupportedLocale
        ]
    
    def export_translations(self, locale: SupportedLocale, output_file: str):
        """Export translations for a locale to file"""
        translations = self.translations.get(locale, {})
        export_data = {
            key: entry.value if isinstance(entry, TranslationEntry) else entry
            for key, entry in translations.items()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    def import_translations(self, locale: SupportedLocale, input_file: str):
        """Import translations for a locale from file"""
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if locale not in self.translations:
            self.translations[locale] = {}
        
        for key, value in data.items():
            self.translations[locale][key] = TranslationEntry(
                key=key,
                value=value,
                locale=locale,
                last_updated=datetime.now()
            )


# Global i18n manager instance
i18n = I18nManager()


def _(key: str, **variables) -> str:
    """Convenient translation function"""
    return i18n.translate(key, variables or None)


def _n(key: str, count: int, **variables) -> str:
    """Convenient translation function with pluralization"""
    return i18n.translate(key, variables or None, count=count)


def set_locale(locale: Union[SupportedLocale, str]):
    """Set global locale"""
    i18n.set_locale(locale)


def get_locale() -> SupportedLocale:
    """Get current locale"""
    return i18n.current_locale