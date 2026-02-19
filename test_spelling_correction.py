
#!/usr/bin/env python3
"""
Test script for advanced spelling correction functionality in grammar corrector.
"""

from grammar_corrector import correct_grammar, initialize_corrector

def test_spelling_correction():
    """Test the advanced spelling correction feature with various examples."""
    
    # Initialize the corrector with spell checking enabled
    initialize_corrector(use_transformer=True, use_openai=False, use_google_nlp=False)
    
    # Test cases with spelling errors (including the ones from your app output)
    test_cases = [
        # Basic spelling errors
        "I wnat to go to skool today",           # want -> want, school -> school
        "YESTERDAY I go to the shoop",            # shop -> shop
        "I luv my famly very much",               # love -> love, family -> family
        "Plese help me lern english",             # please -> please, learn -> learn
        "I am vry hapy today",                    # very -> very, happy -> happy
        "She is a gud teecher",                   # good -> good, teacher -> teacher
        "I ned to by some groceris",              # need -> need, buy -> buy, groceries -> groceries
        "He is ver intelijent",                   # very -> very, intelligent -> intelligent
        "The wether is nice today",               # weather -> weather
        "I cant undrstand this",                  # understand -> understand
        
        # Real-world sign language interpretation errors
        "i wnat apple",                           # From your app output
        "ewery",                                  # From your app output
        
        # Common typing errors
        "teh",                                    # the
        "adn",                                    # and
        "whihc",                                  # which
        "becuase",                                # because
        "shuld",                                  # should
        "coudl",                                  # could
        "woudl",                                  # would
        "definately",                             # definitely
        "neccessary",                             # necessary
        "accomodate",                             # accommodate
        
        # Context-aware corrections
        "I wnat to eet an applle",                # want, eat, apple
        "She is a teecher at the skool",          # teacher, school
        "My famly is ver hapy",                   # family, very, happy
        "Plese hlp me lern",                      # please, help, learn
        "I ned to go to the stor",                # need, store
        
        # Mixed case and punctuation
        "Wnat to go to skool?",                   # Want, school
        "I luv my famly!",                        # love, family
        "Plese, hlp me!",                         # Please, help
    ]
    
    print("🔤 Testing Advanced Spelling Correction Feature")
    print("=" * 80)
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"Original:  '{test_text}'")
        
        try:
            corrected = correct_grammar(test_text, use_transformer=False)
            print(f"Corrected: '{corrected}'")
            
            # Check if spelling correction was applied
            if "[Spelling corrected:" in corrected:
                print("✅ Spelling correction applied")
            else:
                print("ℹ️  No spelling correction needed or applied")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 80)
    print("✅ Advanced spelling correction test completed!")
    print("\n📝 Notes:")
    print("- The system uses multiple correction methods in order of preference")
    print("- Method 1: Enhanced spell checker with custom dictionary")
    print("- Method 2: TextDistance similarity matching")
    print("- Method 3: Difflib pattern matching (fallback)")
    print("- Common sign language misspellings are prioritized")

if __name__ == "__main__":
    test_spelling_correction()
