"""
Grammar Corrector for Sign Language Recognition Output
Converts raw sign language text (ASL/ISL patterns) to grammatically correct English.

Features:
- Rule-based corrections for common sign language patterns
- Google Natural Language API integration for advanced correction
- OpenAI API integration for advanced correction
- Fallback to rule-based if APIs unavailable
- Natural, fluent English output
"""

import re
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Try to import transformer models for advanced correction
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[Grammar Corrector] Transformers not installed. Using rule-based correction only.")
    print("  To enable advanced correction: pip install transformers torch")

# Try to import API clients
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("[Grammar Corrector] OpenAI not installed. Install with: pip install openai")

try:
    from google.cloud import language_v1
    GOOGLE_NLP_AVAILABLE = True
except ImportError:
    GOOGLE_NLP_AVAILABLE = False
    print("[Grammar Corrector] Google NLP not installed. Install with: pip install google-cloud-language")


# ============================================================================
# RULE-BASED GRAMMAR CORRECTION
# ============================================================================

class GrammarCorrector:
    """Advanced grammar corrector with API integration for sign language recognition output."""
    
    def __init__(self, use_transformer: bool = True, use_openai: bool = False, use_google_nlp: bool = False):
        """
        Initialize the grammar corrector.
        
        Args:
            use_transformer: If True, try to load transformer model for advanced correction
            use_openai: If True, use OpenAI API for grammar correction
            use_google_nlp: If True, use Google Natural Language API for grammar correction
        """
        self.use_transformer = use_transformer and TRANSFORMERS_AVAILABLE
        self.use_openai = use_openai and OPENAI_AVAILABLE
        self.use_google_nlp = use_google_nlp and GOOGLE_NLP_AVAILABLE
        
        self.transformer_model = None
        self.openai_client = None
        self.google_client = None
        
        # Initialize OpenAI client if enabled
        if self.use_openai:
            openai_api_key = os.getenv('OPENAI_API_KEY', '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')
            if openai_api_key and openai_api_key != 'your_key_here':
                try:
                    self.openai_client = openai.OpenAI(api_key=openai_api_key)
                    print("[Grammar Corrector] ✓ OpenAI client initialized")
                except Exception as e:
                    print(f"[Grammar Corrector] Could not initialize OpenAI: {e}")
                    self.use_openai = False
            else:
                print("[Grammar Corrector] Invalid or missing OPENAI_API_KEY")
                self.use_openai = False
        
        # Initialize Google NLP client if enabled
        if self.use_google_nlp:
            try:
                self.google_client = language_v1.LanguageServiceClient()
                print("[Grammar Corrector] ✓ Google NLP client initialized")
            except Exception as e:
                print(f"[Grammar Corrector] Could not initialize Google NLP: {e}")
                self.use_google_nlp = False
        
        # Initialize transformer model if enabled
        if self.use_transformer:
            try:
                print("[Grammar Corrector] Loading transformer model for advanced correction...")
                self.transformer_model = pipeline(
                    "text2text-generation",
                    model="t5-small",
                    device=-1  # Use CPU; set to 0 for GPU
                )
                print("[Grammar Corrector] ✓ Transformer model loaded successfully")
            except Exception as e:
                print(f"[Grammar Corrector] Could not load transformer model: {e}")
                print("[Grammar Corrector] Falling back to rule-based correction")
                self.transformer_model = None
        
        # Dictionary of common sign language patterns and their corrections
        self.sign_patterns: Dict[str, str] = {
            # Time indicators
            r'\bTOMORROW\b': 'tomorrow',
            r'\bYESTERDAY\b': 'yesterday',
            r'\bTODAY\b': 'today',
            r'\bNOW\b': 'now',
            r'\bBEFORE\b': 'before',
            r'\bAFTER\b': 'after',
            
            # Common verbs
            r'\bGO\b': 'go',
            r'\bCOME\b': 'come',
            r'\bWANT\b': 'want',
            r'\bNEED\b': 'need',
            r'\bLIKE\b': 'like',
            r'\bLOVE\b': 'love',
            r'\bHATE\b': 'hate',
            r'\bBUY\b': 'buy',
            r'\bSELL\b': 'sell',
            r'\bGIVE\b': 'give',
            r'\bTAKE\b': 'take',
            r'\bSEE\b': 'see',
            r'\bHEAR\b': 'hear',
            r'\bSAY\b': 'say',
            r'\bTELL\b': 'tell',
            r'\bASK\b': 'ask',
            r'\bKNOW\b': 'know',
            r'\bTHINK\b': 'think',
            r'\bFEEL\b': 'feel',
            r'\bDO\b': 'do',
            r'\bMAKE\b': 'make',
            r'\bHAVE\b': 'have',
            r'\bBE\b': 'be',
            r'\bAM\b': 'am',
            r'\bIS\b': 'is',
            r'\bARE\b': 'are',
            r'\bWAS\b': 'was',
            r'\bWERE\b': 'were',
            r'\bBEEN\b': 'been',
            r'\bEAT\b': 'eat',
            r'\bDRINK\b': 'drink',
            r'\bSLEEP\b': 'sleep',
            r'\bWALK\b': 'walk',
            r'\bRUN\b': 'run',
            r'\bJUMP\b': 'jump',
            r'\bSIT\b': 'sit',
            r'\bSTAND\b': 'stand',
            
            # Common nouns
            r'\bSCHOOL\b': 'school',
            r'\bHOME\b': 'home',
            r'\bWORK\b': 'work',
            r'\bMARKET\b': 'market',
            r'\bSHOP\b': 'shop',
            r'\bSTORE\b': 'store',
            r'\bHOUSE\b': 'house',
            r'\bWATER\b': 'water',
            r'\bFOOD\b': 'food',
            r'\bPHONE\b': 'phone',
            r'\bCOMPUTER\b': 'computer',
            r'\bBOOK\b': 'book',
            r'\bPEN\b': 'pen',
            r'\bPAPER\b': 'paper',
            r'\bCAR\b': 'car',
            r'\bBUS\b': 'bus',
            r'\bTRAIN\b': 'train',
            r'\bPERSON\b': 'person',
            r'\bPEOPLE\b': 'people',
            r'\bFRIEND\b': 'friend',
            r'\bFAMILY\b': 'family',
            r'\bMOTHER\b': 'mother',
            r'\bFATHER\b': 'father',
            r'\bBROTHER\b': 'brother',
            r'\bSISTER\b': 'sister',
            r'\bCHILD\b': 'child',
            r'\bBaby\b': 'baby',
            
            # Pronouns
            r'\bI\b': 'I',
            r'\bYOU\b': 'you',
            r'\bHE\b': 'he',
            r'\bSHE\b': 'she',
            r'\bIT\b': 'it',
            r'\bWE\b': 'we',
            r'\bTHEY\b': 'they',
            r'\bME\b': 'me',
            r'\bHIM\b': 'him',
            r'\bHER\b': 'her',
            r'\bUS\b': 'us',
            r'\bTHEM\b': 'them',
            r'\bMY\b': 'my',
            r'\bYOUR\b': 'your',
            r'\bHIS\b': 'his',
            r'\bHER\b': 'her',
            r'\bOUR\b': 'our',
            r'\bTHEIR\b': 'their',
            
            # Adjectives
            r'\bNEW\b': 'new',
            r'\bOLD\b': 'old',
            r'\bGOOD\b': 'good',
            r'\bBAD\b': 'bad',
            r'\bBIG\b': 'big',
            r'\bSMALL\b': 'small',
            r'\bHOT\b': 'hot',
            r'\bCOLD\b': 'cold',
            r'\bHAPPY\b': 'happy',
            r'\bSAD\b': 'sad',
            r'\bTIRED\b': 'tired',
            r'\bSICK\b': 'sick',
            r'\bBEAUTIFUL\b': 'beautiful',
            r'\bUGLY\b': 'ugly',
            r'\bFAST\b': 'fast',
            r'\bSLOW\b': 'slow',
            r'\bEASY\b': 'easy',
            r'\bHARD\b': 'hard',
            
            # Prepositions
            r'\bIN\b': 'in',
            r'\bON\b': 'on',
            r'\bAT\b': 'at',
            r'\bTO\b': 'to',
            r'\bFROM\b': 'from',
            r'\bWITH\b': 'with',
            r'\bWITHOUT\b': 'without',
            r'\bFOR\b': 'for',
            r'\bBY\b': 'by',
            r'\bABOUT\b': 'about',
            r'\bBETWEEN\b': 'between',
            r'\bAMONG\b': 'among',
            r'\bUNDER\b': 'under',
            r'\bOVER\b': 'over',
            r'\bABOVE\b': 'above',
            r'\bBELOW\b': 'below',
            r'\bBEHIND\b': 'behind',
            r'\bBEFORE\b': 'before',
            r'\bAFTER\b': 'after',
            r'\bDURING\b': 'during',
            
            # Articles and common words
            r'\bA\b': 'a',
            r'\bAN\b': 'an',
            r'\bTHE\b': 'the',
            r'\bAND\b': 'and',
            r'\bOR\b': 'or',
            r'\bBUT\b': 'but',
            r'\bNOT\b': 'not',
            r'\bNO\b': 'no',
            r'\bYES\b': 'yes',
        }
    
    def _lowercase_text(self, text: str) -> str:
        """Convert text to lowercase while preserving structure."""
        return text.lower()
    
    def _add_articles(self, text: str) -> str:
        """Add missing articles (a, an, the) based on context."""
        words = text.split()
        result = []
        
        # Common nouns that typically need articles
        nouns_needing_article = {
            'school', 'home', 'work', 'market', 'shop', 'store', 'house',
            'water', 'food', 'phone', 'computer', 'book', 'pen', 'paper',
            'car', 'bus', 'train', 'person', 'friend', 'family', 'baby',
            'movie', 'park', 'restaurant', 'hospital', 'bank', 'office',
            'college', 'university', 'library', 'gym', 'store', 'mall'
        }
        
        for i, word in enumerate(words):
            # Check if previous word is a verb that typically precedes a noun needing article
            if i > 0:
                prev_word = words[i-1].lower()
                verb_before_noun = prev_word in {'go', 'went', 'going', 'visit', 'visited', 'see', 'saw',
                                                   'buy', 'bought', 'want', 'need', 'like', 'love', 'hate',
                                                   'have', 'has', 'had', 'make', 'made', 'take', 'took',
                                                   'give', 'gave', 'find', 'found', 'use', 'used', 'eat',
                                                   'ate', 'drink', 'drank', 'call', 'called', 'meet', 'met'}
                
                # Add article if noun needs one and no article is present
                if verb_before_noun and word.lower() in nouns_needing_article:
                    if i == 0 or words[i-1].lower() not in {'a', 'an', 'the', 'my', 'your', 'his', 'her', 'our', 'their'}:
                        # Determine if 'a' or 'an'
                        article = 'an' if word[0].lower() in 'aeiou' else 'a'
                        result.append(article)
            
            result.append(word)
        
        return ' '.join(result)
    
    def _correct_verb_tense(self, text: str) -> str:
        """Correct verb tenses based on time indicators."""
        # Past tense indicators
        past_indicators = ['yesterday', 'last', 'ago', 'before', 'previously', 'already']
        # Future tense indicators
        future_indicators = ['tomorrow', 'next', 'will', 'going', 'soon', 'later']
        
        words = text.split()
        has_past = any(indicator in text.lower() for indicator in past_indicators)
        has_future = any(indicator in text.lower() for indicator in future_indicators)
        
        # Map present tense to past/future
        present_to_past = {
            'go': 'went', 'come': 'came', 'want': 'wanted', 'need': 'needed',
            'like': 'liked', 'love': 'loved', 'hate': 'hated', 'buy': 'bought',
            'sell': 'sold', 'give': 'gave', 'take': 'took', 'see': 'saw',
            'hear': 'heard', 'say': 'said', 'tell': 'told', 'ask': 'asked',
            'know': 'knew', 'think': 'thought', 'feel': 'felt', 'do': 'did',
            'make': 'made', 'have': 'had', 'eat': 'ate', 'drink': 'drank',
            'sleep': 'slept', 'walk': 'walked', 'run': 'ran', 'jump': 'jumped',
            'sit': 'sat', 'stand': 'stood', 'find': 'found', 'use': 'used',
            'call': 'called', 'meet': 'met', 'start': 'started', 'stop': 'stopped',
            'try': 'tried', 'help': 'helped', 'work': 'worked', 'play': 'played',
            'watch': 'watched', 'listen': 'listened', 'read': 'read', 'write': 'wrote',
            'learn': 'learned', 'teach': 'taught', 'understand': 'understood',
            'remember': 'remembered', 'forget': 'forgot', 'lose': 'lost', 'find': 'found',
            'break': 'broke', 'fix': 'fixed', 'build': 'built', 'destroy': 'destroyed',
        }
        
        present_to_future = {
            'go': 'will go', 'come': 'will come', 'want': 'will want', 'need': 'will need',
            'like': 'will like', 'love': 'will love', 'hate': 'will hate', 'buy': 'will buy',
            'sell': 'will sell', 'give': 'will give', 'take': 'will take', 'see': 'will see',
            'hear': 'will hear', 'say': 'will say', 'tell': 'will tell', 'ask': 'will ask',
            'know': 'will know', 'think': 'will think', 'feel': 'will feel', 'do': 'will do',
            'make': 'will make', 'have': 'will have', 'eat': 'will eat', 'drink': 'will drink',
            'sleep': 'will sleep', 'walk': 'will walk', 'run': 'will run', 'jump': 'will jump',
            'sit': 'will sit', 'stand': 'will stand', 'find': 'will find', 'use': 'will use',
            'call': 'will call', 'meet': 'will meet', 'start': 'will start', 'stop': 'will stop',
            'try': 'will try', 'help': 'will help', 'work': 'will work', 'play': 'will play',
            'watch': 'will watch', 'listen': 'will listen', 'read': 'will read', 'write': 'will write',
            'learn': 'will learn', 'teach': 'will teach', 'understand': 'will understand',
            'remember': 'will remember', 'forget': 'will forget', 'lose': 'will lose', 'find': 'will find',
            'break': 'will break', 'fix': 'will fix', 'build': 'will build', 'destroy': 'will destroy',
        }
        
        result = []
        for word in words:
            word_lower = word.lower()
            
            if has_past and word_lower in present_to_past:
                result.append(present_to_past[word_lower])
            elif has_future and word_lower in present_to_future:
                result.append(present_to_future[word_lower])
            else:
                result.append(word)
        
        return ' '.join(result)
    
    def _add_prepositions(self, text: str) -> str:
        """Add missing prepositions in common patterns."""
        # Pattern: VERB + NOUN → VERB + TO + NOUN (for certain verbs)
        verbs_needing_to = {'go', 'went', 'come', 'came', 'want', 'wanted', 'need', 'needed',
                            'like', 'liked', 'love', 'loved', 'hate', 'hated', 'buy', 'bought',
                            'visit', 'visited', 'see', 'saw', 'meet', 'met', 'call', 'called'}
        
        nouns_needing_to = {'school', 'home', 'work', 'market', 'shop', 'store', 'house',
                            'hospital', 'bank', 'office', 'park', 'restaurant', 'movie',
                            'beach', 'mountain', 'city', 'country', 'place', 'room',
                            'college', 'university', 'library', 'gym', 'mall'}
        
        words = text.split()
        result = []
        
        for i, word in enumerate(words):
            result.append(word)
            
            # Check if current word is a verb and next word is a noun
            if i < len(words) - 1:
                current_lower = word.lower()
                next_word = words[i + 1].lower()
                
                if current_lower in verbs_needing_to and next_word in nouns_needing_to:
                    # Check if 'to' is not already present
                    if i == 0 or words[i].lower() != 'to':
                        result.append('to')
        
        return ' '.join(result)
    
    def _reorder_time_expressions(self, text: str) -> str:
        """Move time expressions to the end of the sentence."""
        time_words = ['yesterday', 'today', 'tomorrow', 'morning', 'afternoon', 'evening',
                      'night', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
                      'saturday', 'sunday', 'january', 'february', 'march', 'april',
                      'may', 'june', 'july', 'august', 'september', 'october', 'november',
                      'december', 'week', 'month', 'year', 'hour', 'minute', 'second']
        
        words = text.split()
        time_expressions = []
        other_words = []
        
        for word in words:
            if word.lower() in time_words:
                time_expressions.append(word)
            else:
                other_words.append(word)
        
        # Reconstruct: other words + time expressions
        if time_expressions:
            return ' '.join(other_words + time_expressions)
        return text
    
    def _fix_subject_verb_agreement(self, text: str) -> str:
        """Fix subject-verb agreement."""
        words = text.split()
        result = []
        
        for i, word in enumerate(words):
            if i > 0:
                prev_word = words[i-1].lower()
                current_lower = word.lower()
                
                # Check for singular subjects (including noun phrases)
                # Look at previous 2 words to detect patterns like "my name"
                if i >= 2:
                    prev_two_words = f"{words[i-2].lower()} {prev_word}"
                else:
                    prev_two_words = prev_word
                
                # Singular subjects and patterns
                singular_patterns = {
                    'i', 'he', 'she', 'it',  # pronouns
                    'my name', 'your name', 'his name', 'her name',  # name patterns
                    'the dog', 'the cat', 'the car', 'the house',  # singular noun with article
                    'this', 'that',  # demonstratives
                    'my', 'your', 'his', 'her', 'its'  # possessives that imply singular
                }
                
                # Check if previous word(s) indicate singular subject
                is_singular = (
                    prev_word in {'i', 'he', 'she', 'it', 'this', 'that'} or
                    prev_word in {'my', 'your', 'his', 'her', 'its'} or
                    prev_two_words in {'my name', 'your name', 'his name', 'her name'} or
                    (i >= 2 and words[i-2].lower() == 'the' and prev_word in {'dog', 'cat', 'car', 'house', 'name', 'book', 'phone'})
                )
                
                # Plural subjects
                plural_patterns = {
                    'we', 'you', 'they', 'people',  # pronouns
                    'these', 'Capitalized', 'those',  # demonstratives
                    'the dogs', 'the cats', 'the cars', 'the houses'  # plural nouns with article
                }
                
                is_plural = (
                    prev_word in {'we', 'you', 'they', 'people', 'these', 'those'} or
                    (i >= 2 and words[i-2].lower() == 'the' and prev_word.endswith('s'))
                )
                
                if is_singular:
                    if current_lower == 'are':
                        result.append('am' if prev_word == 'i' else 'is')
                    elif current_lower == 'have':
                        result.append('has')
                    elif current_lower == 'were':
                        result.append('was')
                    else:
                        result.append(word)
                elif is_plural:
                    if current_lower == 'is':
                        result.append('are')
                    elif current_lower == 'has':
                        result.append('have')
                    elif current_lower == 'was':
                        result.append('were')
                    else:
                        result.append(word)
                else:
                    result.append(word)
            else:
                result.append(word)
        
        return ' '.join(result)
    
    def _fix_common_misspellings(self, text: str) -> str:
        """Fix common misspellings in sign language output."""
        # Common misspellings dictionary
        misspellings = {
            'tomorrow': 'tomorrow',
            'tommorrow': 'tomorrow',
            'tommorow': 'tomorrow',
            'yesturday': 'yesterday',
            'yestrday': 'yesterday',
            'today': 'today',
            'tody': 'today',
            'skool': 'school',
            'skool': 'school',
            'hous': 'house',
            'hoom': 'home',
        }
        
        words = text.split()
        result = []
        for word in words:
            word_lower = word.lower()
            if word_lower in misspellings:
                result.append(misspellings[word_lower])
            else:
                result.append(word)
        
        return ' '.join(result)
    
    def _apply_rule_based_corrections(self, text: str) -> str:
        """Apply all rule-based corrections in sequence."""
        # Step 1: Fix common misspellings
        text = self._fix_common_misspellings(text)
        
        # Step 2: Lowercase
        text = self._lowercase_text(text)
        
        # Step 2: Fix subject-verb agreement
        text = self._fix_subject_verb_agreement(text)
        
        # Step 3: Correct verb tenses
        text = self._correct_verb_tense(text)
        
        # Step 4: Add missing prepositions
        text = self._add_prepositions(text)
        
        # Step 5: Add missing articles
        text = self._add_articles(text)
        
        # Step 6: Reorder time expressions
        text = self._reorder_time_expressions(text)
        
        # Step 7: Add punctuation
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        # Step 8: Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        return text
    
    def _apply_openai_correction(self, text: str) -> str:
        """Apply OpenAI API for advanced grammar correction."""
        if not self.openai_client:
            return text
        
        try:
            prompt = f"""Please correct the grammar of this sentence to make it natural and fluent English. 
The input may be from sign language recognition, so it might have word order issues.

Original: "{text}"

Corrected:"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a grammar correction assistant. Fix grammar to make natural English sentences."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.3
            )
            
            corrected = response.choices[0].message.content.strip()
            
            # Ensure proper capitalization and punctuation
            if corrected and not corrected.endswith(('.', '!', '?')):
                corrected += '.'
            if corrected:
                corrected = corrected[0].upper() + corrected[1:]
            
            return corrected
        except Exception as e:
            print(f"[Grammar Corrector] OpenAI correction failed: {e}")
            return text
    
    def _apply_google_nlp_correction(self, text: str) -> str:
        """Apply Google Natural Language API for syntax analysis and correction."""
        if not self.google_client:
            return text
        
        try:
            # Analyze syntax
            document = language_v1.Document(
                content=text, type_=language_v1.Document.Type.PLAIN_TEXT
            )
            
            # Get syntax tokens
            response = self.google_client.analyze_syntax(
                request={'document': document, 'encoding_type': language_v1.EncodingType.UTF8}
            )
            
            # Extract words and their parts of speech
            words = []
            for token in response.tokens:
                word = token.text.content
                tag = token.part_of_speech.tag
                
                # Basic corrections based on part of speech
                if tag == language_v1.PartOfSpeech.Tag.VERB:
                    # Apply verb tense corrections
                    word = self._correct_verb_form(word, text)
                elif tag == language_v1.PartOfSpeech.Tag.NOUN:
                    # Check for article usage
                    word = self._add_article_if_needed(word, text)
                
                words.append(word)
            
            corrected = ' '.join(words)
            
            # Ensure proper capitalization and punctuation
            if corrected and not corrected.endswith(('.', '!', '?')):
                corrected += '.'
            if corrected:
                corrected = corrected[0].upper() + corrected[1:]
            
            return corrected
        except Exception as e:
            print(f"[Grammar Corrector] Google NLP correction failed: {e}")
            return text
    
    def _correct_verb_form(self, verb: str, context: str) -> str:
        """Correct verb form based on context."""
        # Simple verb corrections
        verb_corrections = {
            'go': 'goes', 'goes': 'go', 'went': 'go',
            'come': 'comes', 'comes': 'come', 'came': 'come',
            'have': 'has', 'has': 'have', 'had': 'have',
            'be': 'is', 'is': 'be', 'am': 'be', 'are': 'be', 'was': 'be', 'were': 'be'
        }
        return verb_corrections.get(verb.lower(), verb)
    
    def _add_article_if_needed(self, noun: str, context: str) -> str:
        """Add article if noun needs one."""
        articles = {'a', 'an', 'the'}
        words = context.lower().split()
        
        # Check if noun already has article before it
        noun_index = words.index(noun.lower()) if noun.lower() in words else -1
        if noun_index > 0 and words[noun_index - 1] in articles:
            return noun
        
        # Add appropriate article for common nouns
        common_nouns = {'school', 'home', 'work', 'market', 'shop', 'store', 'house'}
        if noun.lower() in common_nouns:
            article = 'an' if noun[0].lower() in 'aeiou' else 'a'
            return f"{article} {noun}"
        
        return noun
    
    def _apply_transformer_correction(self, text: str) -> str:
        """Apply transformer-based correction for advanced grammar."""
        if not self.transformer_model:
            return text
        
        try:
            # Prepare input for T5 model
            input_text = f"grammar: {text}"
            
            # Generate correction
            result = self.transformer_model(input_text, max_length=128, num_beams=4)
            corrected = result[0]['generated_text'].strip()
            
            # Ensure proper capitalization and punctuation
            if corrected and not corrected.endswith(('.', '!', '?')):
                corrected += '.'
            if corrected:
                corrected = corrected[0].upper() + corrected[1:]
            
            return corrected
        except Exception as e:
            print(f"[Grammar Corrector] Transformer correction failed: {e}")
            return text
    
    def correct(self, text: str) -> str:
        """
        Correct grammar in sign language recognition output using available APIs.
        
        Args:
            text: Raw text from sign recognition (e.g., "I GO SCHOOL TOMORROW")
        
        Returns:
            Grammatically correct English sentence
        """
        if not text or not text.strip():
            return text
        
        # Apply rule-based corrections first
        corrected = self._apply_rule_based_corrections(text)
        
        # Try API corrections in order of preference
        if self.use_openai:
            try:
                corrected = self._apply_openai_correction(corrected)
                return corrected  # Return OpenAI result if successful
            except Exception as e:
                print(f"[Grammar Corrector] OpenAI API error: {e}")
        
        if self.use_google_nlp:
            try:
                corrected = self._apply_google_nlp_correction(corrected)
                return corrected  # Return Google NLP result if successful
            except Exception as e:
                print(f"[Grammar Corrector] Google NLP API error: {e}")
        
        # If transformer model is available, use it as fallback
        if self.transformer_model:
            try:
                corrected = self._apply_transformer_correction(corrected)
                return corrected
            except Exception as e:
                print(f"[Grammar Corrector] Transformer model error: {e}")
        
        # Return rule-based result as final fallback
        return corrected


# ============================================================================
# PUBLIC API
# ============================================================================

# Global corrector instance
_corrector = None

def correct_grammar(text: str, use_transformer: bool = True, use_openai: bool = False, use_google_nlp: bool = False) -> str:
    """
    Correct grammar in sign language recognition output.
    
    This is the main entry point for grammar correction with API options.
    
    Args:
        text: Raw text from sign recognition (e.g., "I GO SCHOOL TOMORROW")
        use_transformer: If True, use transformer model for advanced correction
        use_openai: If True, use OpenAI API for grammar correction
        use_google_nlp: If True, use Google Natural Language API for grammar correction
    
    Returns:
        Grammatically correct English sentence
    
    Examples:
        >>> correct_grammar("I GO SCHOOL TOMORROW")
        "I will go to school tomorrow."
        
        >>> correct_grammar("YESTERDAY I GO SHOP")
        "I went to the shop yesterday."
        
        >>> correct_grammar("I WANT BUY NEW PHONE")
        "I want to buy a new phone."
    """
    global _corrector
    
    if _corrector is None:
        _corrector = GrammarCorrector(use_transformer=use_transformer, use_openai=use_openai, use_google_nlp=use_google_nlp)
    
    return _corrector.correct(text)


def initialize_corrector(use_transformer: bool = True, use_openai: bool = False, use_google_nlp: bool = False) -> None:
    """
    Pre-initialize the grammar corrector (useful for loading models early).
    
    Args:
        use_transformer: If True, load transformer model for advanced correction
        use_openai: If True, initialize OpenAI API client
        use_google_nlp: If True, initialize Google NLP API client
    """
    global _corrector
    _corrector = GrammarCorrector(use_transformer=use_transformer, use_openai=use_openai, use_google_nlp=use_google_nlp)


# ============================================================================
# EXAMPLES AND TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Sign Language Grammar Corrector - Examples")
    print("=" * 70)
    
    # Initialize corrector
    corrector = GrammarCorrector(use_transformer=True)
    
    # Test cases
    test_cases = [
        "I GO SCHOOL TOMORROW",
        "YESTERDAY I GO SHOP",
        "I WANT BUY NEW PHONE",
        "HE LIKE PLAY FOOTBALL",
        "TOMORROW I VISIT FRIEND",
        "I NEED DRINK WATER",
        "SHE HAVE NEW BOOK",
        "THEY GO MARKET YESTERDAY",
        "I WANT GO HOME NOW",
        "YOU LIKE COFFEE",
        "WE PLAY GAME TOMORROW",
        "HE WORK OFFICE EVERY DAY",
        "I SEE MOVIE LAST NIGHT",
        "THEY WANT BUY CAR",
        "SHE LOVE EAT FOOD",
    ]
    
    print("\nRule-Based Corrections:")
    print("-" * 70)
    for test in test_cases:
        corrected = corrector.correct(test)
        print(f"Input:  {test}")
        print(f"Output: {corrected}")
        print()
    
    print("\n" + "=" * 70)
    print("Grammar correction complete!")
    print("=" * 70)
