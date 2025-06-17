 #!/usr/bin/env python3
"""
DSPy-based chunk quality classifier using Gemini.
Directly classifies academic paper chunks as keep or skip.
"""

import dspy
import os
import warnings
import logging
from typing import Literal
from dotenv import load_dotenv

# Suppress noisy warnings and logs
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
os.environ["LITELLM_LOG"] = "ERROR"

# Load environment variables
load_dotenv()

# Verify API key is available
google_api_key = os.getenv('GOOGLE_API_KEY')
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

# Configure DSPy with Gemini using the correct LiteLLM format
# For Google AI Studio, the format should be 'gemini/model-name'
lm = dspy.LM('gemini/gemini-1.5-flash', api_key=google_api_key)
dspy.configure(lm=lm)

class ChunkQualitySignature(dspy.Signature):
    """Simple signature for chunk quality assessment."""
    chunk_text: str = dspy.InputField(desc="The text chunk to assess")
    classification: str = dspy.OutputField(desc="Classification: either 'keep' or 'skip'")
    confidence: float = dspy.OutputField(desc="Confidence score between 0 and 1")

class ChunkClassifier(dspy.Module):
    """DSPy module for simple chunk classification."""
    
    def __init__(self):
        super().__init__()
        self.classifier = dspy.ChainOfThought(ChunkQualitySignature)
    
    def forward(self, chunk_text: str):
        """Classify chunk as keep or skip."""
        return self.classifier(chunk_text=chunk_text)

def create_optimized_classifier(trainset=None):
    """Create and optionally optimize the classifier."""
    classifier = ChunkClassifier()
    
    if trainset:
        # Define optimization metric
        def accuracy_metric(pred, gold, trace=None):
            return pred.classification == gold.classification
        
        # Optimize using DSPy's BootstrapFewShot
        optimizer = dspy.BootstrapFewShot(
            metric=accuracy_metric,
            max_bootstrapped=50,
            max_labeled=20,
            num_threads=4
        )
        
        # Compile optimized version
        classifier = optimizer.compile(classifier, trainset=trainset)
    
    return classifier

# Example usage
if __name__ == "__main__":
    classifier = create_optimized_classifier()
    
    # Test with a good chunk
    good_chunk = """
    The results demonstrate a significant correlation (p < 0.001) between sleep
    duration and cognitive performance across all age groups. Participants who
    maintained 7-9 hours of sleep consistently showed improved memory retention
    and faster reaction times compared to those with irregular sleep patterns.
    """
    
    result = classifier(chunk_text=good_chunk)
    print(f"Good chunk classification: {result.classification}")
    print(f"Confidence: {result.confidence:.2f}")
    
    # Test with a bad chunk
    bad_chunk = """
    Author manuscript
    Page 2
    NIH-PA Author Manuscript
    """
    
    result = classifier(chunk_text=bad_chunk)
    print(f"Bad chunk classification: {result.classification}")
    print(f"Confidence: {result.confidence:.2f}")