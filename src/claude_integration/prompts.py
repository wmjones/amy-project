"""
Prompt templates for different document types.
"""

from typing import Dict, Any
import json


class PromptTemplates:
    """Collection of prompt templates for document analysis."""

    @staticmethod
    def get_default_analysis_prompt() -> str:
        """Get default analysis prompt template."""
        return """Analyze this document and extract the following information in JSON format:

{
    "document_type": "type of document (invoice, contract, letter, report, etc.)",
    "category": "general category (financial, legal, personal, work, medical, etc.)",
    "date": "document date if available (YYYY-MM-DD format)",
    "subjects": ["list", "of", "main", "topics"],
    "entities": {
        "people": ["names of people mentioned"],
        "organizations": ["company or organization names"],
        "locations": ["places mentioned"]
    },
    "summary": "brief summary of document content",
    "suggested_folder": "path/to/suggested/folder",
    "confidence_score": 0.95,
    "tags": ["relevant", "tags", "for", "organization"]
}

Document content:
{content}"""

    @staticmethod
    def get_image_analysis_prompt() -> str:
        """Get prompt for analyzing document images."""
        return """This is an image of a document. Please analyze the visible text and structure to extract the following information in JSON format:

{
    "document_type": "type of document based on visual structure",
    "visible_text": "main text content visible in the image",
    "date": "any visible date (YYYY-MM-DD format)",
    "category": "general category based on content",
    "quality": "image quality (clear, blurry, partial)",
    "suggested_folder": "path/to/suggested/folder",
    "confidence_score": 0.85,
    "requires_ocr": true/false,
    "summary": "brief description of what the document appears to be"
}

Please analyze the document image and provide the requested information."""

    @staticmethod
    def get_pdf_analysis_prompt() -> str:
        """Get prompt for analyzing PDF documents."""
        return """Analyze this PDF document and extract structured information for organization. Pay special attention to:
- Document headers and titles
- Dates and version numbers
- Author information
- Document structure (sections, chapters)

Provide the analysis in JSON format:

{
    "document_type": "specific type of PDF document",
    "title": "document title if available",
    "author": "author or creator",
    "date": "creation or modification date (YYYY-MM-DD)",
    "page_count": number,
    "category": "document category",
    "subjects": ["main", "topics", "covered"],
    "table_of_contents": ["section", "titles", "if", "available"],
    "suggested_folder": "path/to/suggested/folder",
    "confidence_score": 0.9,
    "tags": ["relevant", "tags"],
    "summary": "comprehensive summary of the document"
}

PDF Content:
{content}"""

    @staticmethod
    def get_email_analysis_prompt() -> str:
        """Get prompt for analyzing email documents."""
        return """Analyze this email and extract organizational information:

{
    "email_type": "type of email (personal, work, newsletter, etc.)",
    "from": "sender email/name",
    "to": ["recipient", "emails"],
    "subject": "email subject",
    "date": "email date (YYYY-MM-DD)",
    "category": "email category",
    "attachments": ["list", "of", "attachments"],
    "action_required": true/false,
    "priority": "high/medium/low",
    "suggested_folder": "path/to/suggested/folder",
    "tags": ["relevant", "tags"],
    "summary": "brief summary of email content"
}

Email content:
{content}"""

    @staticmethod
    def get_financial_analysis_prompt() -> str:
        """Get prompt for analyzing financial documents."""
        return """Analyze this financial document with special attention to monetary values, dates, and accounts:

{
    "document_type": "specific financial document type",
    "institution": "financial institution name",
    "account_number": "account number (last 4 digits only)",
    "date": "document date (YYYY-MM-DD)",
    "period": "time period covered",
    "amounts": {
        "total": 0.00,
        "currency": "USD",
        "transactions": []
    },
    "category": "financial document category",
    "tax_year": "YYYY if applicable",
    "suggested_folder": "financial/year/type",
    "confidence_score": 0.95,
    "tags": ["financial", "tags"],
    "summary": "summary of financial information"
}

Financial document content:
{content}"""

    @staticmethod
    def get_legal_analysis_prompt() -> str:
        """Get prompt for analyzing legal documents."""
        return """Analyze this legal document and extract key information:

{
    "document_type": "specific legal document type",
    "parties": ["list", "of", "parties", "involved"],
    "date": "document date (YYYY-MM-DD)",
    "effective_date": "when it takes effect",
    "expiration_date": "when it expires (if applicable)",
    "jurisdiction": "legal jurisdiction",
    "case_number": "case or reference number",
    "category": "legal category",
    "key_terms": ["important", "legal", "terms"],
    "suggested_folder": "legal/year/type",
    "confidence_score": 0.9,
    "requires_review": true/false,
    "summary": "summary of legal content"
}

Legal document content:
{content}"""

    @staticmethod
    def get_medical_analysis_prompt() -> str:
        """Get prompt for analyzing medical documents."""
        return """Analyze this medical document while respecting privacy:

{
    "document_type": "type of medical document",
    "date": "document date (YYYY-MM-DD)",
    "provider": "healthcare provider/institution",
    "category": "medical category",
    "document_class": "report/prescription/test/etc",
    "follow_up_required": true/false,
    "suggested_folder": "medical/year/type",
    "confidence_score": 0.85,
    "tags": ["medical", "tags"],
    "summary": "brief medical summary (no personal details)"
}

Medical document content:
{content}"""

    @staticmethod
    def select_prompt(file_type: str, content: str = "") -> str:
        """Select appropriate prompt based on file type or content.

        Args:
            file_type: File extension or MIME type
            content: Optional content preview

        Returns:
            Appropriate prompt template
        """
        # Map file types to prompts
        type_to_prompt = {
            "pdf": PromptTemplates.get_pdf_analysis_prompt(),
            "jpg": PromptTemplates.get_image_analysis_prompt(),
            "jpeg": PromptTemplates.get_image_analysis_prompt(),
            "png": PromptTemplates.get_image_analysis_prompt(),
            "eml": PromptTemplates.get_email_analysis_prompt(),
            "msg": PromptTemplates.get_email_analysis_prompt(),
        }

        # Check content for specific document types
        content_lower = content.lower()
        if any(
            term in content_lower
            for term in ["invoice", "payment", "balance", "amount due"]
        ):
            return PromptTemplates.get_financial_analysis_prompt()
        elif any(
            term in content_lower
            for term in ["agreement", "contract", "hereby", "whereas"]
        ):
            return PromptTemplates.get_legal_analysis_prompt()
        elif any(
            term in content_lower
            for term in ["diagnosis", "prescription", "patient", "medical"]
        ):
            return PromptTemplates.get_medical_analysis_prompt()

        # Return based on file type or default
        return type_to_prompt.get(
            file_type.lower(), PromptTemplates.get_default_analysis_prompt()
        )

    @staticmethod
    def format_prompt(template: str, **kwargs) -> str:
        """Format a prompt template with provided values.

        Args:
            template: Prompt template string
            **kwargs: Values to insert into template

        Returns:
            Formatted prompt
        """
        return template.format(**kwargs)
