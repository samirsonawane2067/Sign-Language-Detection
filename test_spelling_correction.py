#!/usr/bin/env python3
"""
Test script for spelling correction functionality in grammar corrector.
"""

from grammar_corrector import correct_grammar, initialize_corrector

def test_spelling_correction():
    """Test the spelling correction feature with various examples."""
    
    # Initialize the corrector with spell checking enabled
    initialize_corrector(use_transformer=True, use_openai=True, use_google_nlp=False)
    
    # Test cases with spelling errors
    test_cases = [
        "I wnat to go to skool today",  # want -> want, school -> school
        "YESTERDAY I go to the shoop",   # shop -> shop
        "I luv my famly very much",      # love -> love, family -> family
        "Plese help me lern english",    # please -> please, learn -> learn
        "I am vry hapy today",           # very -> very, happy -> happy
        "She is a gud teecher",          # good -> good, teacher -> teacher
        "I ned to by some groceris",     # need -> need, buy -> buy, groceries -> groceries
        "He is ver intelijent",          # very -> very, intelligent -> intelligent
        "The wether is nice today",      # weather -> weather
        "I cant undrstand this",         # understand -> understand
    ]
    
    print("🔤 Testing Spelling Correction Feature")
    print("=" * 60)
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"Original:  {test_text}")
        
        try:
            corrected = correct_grammar(test_text, use_transformer=True, use_openai=False)
            print(f"Corrected: {corrected}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Spelling correction test completed!")

if __name__ == "__main__":
    test_spelling_correction()
