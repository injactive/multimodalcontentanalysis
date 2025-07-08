import json
from src.models.domain import AnalysisRequest, ContentType


def import_testcases(file_str: str) -> list[dict]:
    """
    Import and transform test cases from a JSON file.

    Each test case is converted into a structured dictionary with a name and 
    an AnalysisRequest object. The content type is inferred from the presence 
    of text and/or image data.
    """
    # Load JSON data from file
    with open(file_str, 'r') as file:
        test_cases = json.load(file)

    transformed_test_cases = []

    for case in test_cases:
        # Determine content type based on available fields
        if "text" in case and "image_url" in case:
            content_type = ContentType.MULTIMODAL
        elif "text" in case:
            content_type = ContentType.TEXT
        elif "image_url" in case:
            content_type = ContentType.IMAGE
        else:
            content_type = None  # Fallback or error handling if needed

        # Construct AnalysisRequest object
        request = AnalysisRequest(
            text=case.get("text"),
            image_url=case.get("image_url"),
            content_type=content_type,
            metadata=case.get("metadata"),
        )

        # Wrap in a named dictionary structure
        transformed_test_cases.append({
            "name": case.get("name"),
            "request": request
        })

    return transformed_test_cases
