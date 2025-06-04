import streamlit as st
import google.generativeai as genai
import anthropic
import openai
from enum import Enum
import pandas as pd
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class OMOPSchemaHandler:
    """Handles OMOP CDM schema information and validation"""
    
    def __init__(self):
        # Core OMOP CDM tables with key columns
        self.omop_tables = {
            'person': {
                'primary_key': 'person_id',
                'columns': ['person_id', 'gender_concept_id', 'year_of_birth', 'month_of_birth', 
                          'day_of_birth', 'birth_datetime', 'race_concept_id', 'ethnicity_concept_id'],
                'description': 'Contains demographic information about patients'
            },
            'condition_occurrence': {
                'primary_key': 'condition_occurrence_id',
                'columns': ['condition_occurrence_id', 'person_id', 'condition_concept_id', 
                          'condition_start_date', 'condition_end_date', 'condition_type_concept_id'],
                'description': 'Contains patient conditions and diagnoses'
            },
            'drug_exposure': {
                'primary_key': 'drug_exposure_id',
                'columns': ['drug_exposure_id', 'person_id', 'drug_concept_id', 'drug_exposure_start_date',
                          'drug_exposure_end_date', 'drug_type_concept_id', 'quantity', 'days_supply'],
                'description': 'Contains patient medication exposures'
            },
            'procedure_occurrence': {
                'primary_key': 'procedure_occurrence_id', 
                'columns': ['procedure_occurrence_id', 'person_id', 'procedure_concept_id',
                          'procedure_date', 'procedure_type_concept_id'],
                'description': 'Contains patient procedures'
            },
            'measurement': {
                'primary_key': 'measurement_id',
                'columns': ['measurement_id', 'person_id', 'measurement_concept_id', 'measurement_date',
                          'value_as_number', 'value_as_concept_id', 'unit_concept_id'],
                'description': 'Contains patient measurements and lab results'
            },
            'observation': {
                'primary_key': 'observation_id',
                'columns': ['observation_id', 'person_id', 'observation_concept_id', 'observation_date',
                          'value_as_string', 'value_as_number'],
                'description': 'Contains patient observations'
            },
            'concept': {
                'primary_key': 'concept_id',
                'columns': ['concept_id', 'concept_name', 'domain_id', 'vocabulary_id', 
                          'concept_class_id', 'concept_code'],
                'description': 'Contains standardized concepts and terminologies'
            },
            'concept_ancestor': {
                'primary_key': 'ancestor_concept_id, descendant_concept_id',
                'columns': ['ancestor_concept_id', 'descendant_concept_id', 'min_levels_of_separation',
                          'max_levels_of_separation'],
                'description': 'Contains hierarchical relationships between concepts - CRITICAL for proper OMOP queries'
            },
            'concept_relationship': {
                'primary_key': 'concept_id_1, concept_id_2, relationship_id',
                'columns': ['concept_id_1', 'concept_id_2', 'relationship_id', 'valid_start_date',
                          'valid_end_date'],
                'description': 'Contains relationships between concepts (maps-to, subsumes, etc.)'
            },
            'concept_synonym': {
                'primary_key': 'concept_id, concept_synonym_name',
                'columns': ['concept_id', 'concept_synonym_name', 'language_concept_id'],
                'description': 'Contains alternative names and synonyms for concepts'
            }
        }
        
        # Common medical concepts for examples with their hierarchical relationships
        self.sample_concepts = {
            'diabetes': {
                'parent_concept_id': 201820,  # Diabetes mellitus
                'example_descendants': [4087682, 4048852, 443767, 443729],
                'description': 'Diabetes and all subtypes'
            },
            'hypertension': {
                'parent_concept_id': 316866,  # Hypertensive disorder
                'example_descendants': [320128, 4024552, 432867],
                'description': 'Hypertension and all subtypes'
            },
            'heart_attack': {
                'parent_concept_id': 4329847,  # Myocardial infarction
                'example_descendants': [312327, 444406, 438170],
                'description': 'Heart attack and related conditions'
            },
            'benign_prostatic_hyperplasia': {
                'parent_concept_id': 198803,  # BPH diagnosis (your example!)
                'example_descendants': [194997, 4051466],
                'description': 'Benign prostatic hyperplasia and related conditions'
            }
        }
    
    def get_schema_info(self) -> str:
        """Returns formatted schema information for the LLM"""
        schema_info = "OMOP CDM Schema Information:\n\n"
        
        for table_name, table_info in self.omop_tables.items():
            schema_info += f"Table: {table_name}\n"
            schema_info += f"Description: {table_info['description']}\n"
            schema_info += f"Primary Key: {table_info['primary_key']}\n"
            schema_info += f"Key Columns: {', '.join(table_info['columns'])}\n\n"
        
        schema_info += "\nImportant Notes:\n"
        schema_info += "- All concept IDs reference the 'concept' table for standardized terminology\n"
        schema_info += "- Dates should be in YYYY-MM-DD format\n"
        schema_info += "- Always join with person table using person_id for patient-level queries\n"
        schema_info += "- Use concept table to translate concept names to IDs when needed\n\n"
        
        schema_info += "CRITICAL: Use concept_ancestor for hierarchical queries!\n"
        schema_info += "- To find all patients with diabetes (including subtypes), use:\n"
        schema_info += "  WHERE condition_concept_id IN (\n"
        schema_info += "    SELECT descendant_concept_id FROM concept_ancestor\n"
        schema_info += "    WHERE ancestor_concept_id = 201820 -- Diabetes mellitus\n"
        schema_info += "  )\n"
        schema_info += "- This captures Type 1, Type 2, gestational diabetes, etc.\n"
        schema_info += "- Without concept_ancestor, you miss important clinical relationships!\n"
        
        return schema_info

print("✅ OMOP Schema Handler class defined!")

class DatasetHandler:
    """Handles the custom OMOP query dataset for few-shot learning and evaluation"""
    
    def __init__(self):
        self.dataset = None
        self.vectorizer = None
        self.query_vectors = None
    
    def load_dataset(self, csv_file_or_path) -> bool:
        """Load the CSV dataset with natural language and SQL pairs"""
        try:
            # Handle both file path and uploaded file object
            if hasattr(csv_file_or_path, 'read'):
                # It's an uploaded file object - try multiple encodings
                csv_file_or_path.seek(0)
                
                # Try different encodings
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
                df = None
                
                for encoding in encodings:
                    try:
                        csv_file_or_path.seek(0)
                        content = csv_file_or_path.read().decode(encoding)
                        
                        # Convert to StringIO for pandas
                        from io import StringIO
                        df = pd.read_csv(StringIO(content))
                        print(f"✅ Successfully read file with {encoding} encoding")
                        break
                        
                    except (UnicodeDecodeError, UnicodeError):
                        print(f"⚠️ Failed to read with {encoding} encoding, trying next...")
                        continue
                    except Exception as e:
                        print(f"⚠️ Error with {encoding}: {str(e)}")
                        continue
                
                if df is None:
                    print("❌ Could not read file with any supported encoding")
                    return False
            else:
                # It's a file path
                # Try different encodings for file path
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                df = None
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(csv_file_or_path, encoding=encoding)
                        print(f"✅ Successfully read file with {encoding} encoding")
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                    except Exception as e:
                        print(f"⚠️ Error with {encoding}: {str(e)}")
                        continue
                
                if df is None:
                    print("❌ Could not read file with any supported encoding")
                    return False
            
            print(f"📊 Original dataset shape: {df.shape}")
            print(f"📋 Original columns: {list(df.columns)}")
            
            # Show first few rows for debugging (safely handle any encoding issues in display)
            print("📝 First few rows:")
            try:
                print(df.head(2))
            except Exception as e:
                print(f"⚠️ Could not display rows: {str(e)}")
            
            # Auto-detect column names (flexible naming)
            possible_nl_cols = ['natural_language', 'nl_query', 'question', 'query', 'natural_query', 'input', 'text']
            possible_sql_cols = ['sql', 'sql_query', 'omop_sql', 'answer', 'sql_answer', 'output', 'target']
            
            nl_col = None
            sql_col = None
            
            # Try exact matches first
            for col in df.columns:
                try:
                    col_lower = str(col).lower().strip()
                    if col_lower in possible_nl_cols:
                        nl_col = col
                    elif col_lower in possible_sql_cols:
                        sql_col = col
                except:
                    continue
            
            # If exact match fails, try partial matches
            if nl_col is None:
                for col in df.columns:
                    try:
                        col_lower = str(col).lower().strip()
                        if any(keyword in col_lower for keyword in ['question', 'query', 'natural', 'input']):
                            nl_col = col
                            break
                    except:
                        continue
            
            if sql_col is None:
                for col in df.columns:
                    try:
                        col_lower = str(col).lower().strip()
                        if any(keyword in col_lower for keyword in ['sql', 'answer', 'output', 'target']):
                            sql_col = col
                            break
                    except:
                        continue
            
            # If auto-detection fails, use first two columns
            if nl_col is None or sql_col is None:
                columns = df.columns.tolist()
                if len(columns) < 2:
                    print(f"❌ Dataset needs at least 2 columns, found {len(columns)}")
                    return False
                    
                nl_col = columns[0]
                sql_col = columns[1]
                print(f"⚠️ Auto-detection failed, using columns: '{nl_col}' and '{sql_col}'")
            else:
                print(f"✅ Detected columns: '{nl_col}' (natural language) and '{sql_col}' (SQL)")
            
            # Create the cleaned dataset
            self.dataset = df[[nl_col, sql_col]].copy()
            self.dataset.columns = ['natural_language', 'sql_query']
            
            print(f"📊 Before cleaning: {len(self.dataset)} rows")
            
            # Clean the data more robustly
            initial_size = len(self.dataset)
            
            # Convert to string and handle NaN values
            self.dataset['natural_language'] = self.dataset['natural_language'].astype(str)
            self.dataset['sql_query'] = self.dataset['sql_query'].astype(str)
            
            # Remove rows with 'nan', empty strings, or just whitespace
            self.dataset = self.dataset[
                (self.dataset['natural_language'].str.lower() != 'nan') & 
                (self.dataset['sql_query'].str.lower() != 'nan') &
                (self.dataset['natural_language'].str.strip() != '') & 
                (self.dataset['sql_query'].str.strip() != '')
            ].reset_index(drop=True)
            
            print(f"📊 After cleaning: {len(self.dataset)} rows (removed {initial_size - len(self.dataset)} empty rows)")
            
            if len(self.dataset) == 0:
                print("❌ No valid data rows found after cleaning")
                return False
            
            if len(self.dataset) < 5:
                print(f"⚠️ Warning: Only {len(self.dataset)} valid rows found. Consider adding more data.")
            
            # Create TF-IDF vectors for similarity matching
            try:
                # Clean text for TF-IDF (remove any problematic characters)
                clean_texts = []
                for text in self.dataset['natural_language']:
                    try:
                        # Convert to string and encode/decode to clean up any encoding issues
                        clean_text = str(text).encode('utf-8', errors='ignore').decode('utf-8')
                        clean_texts.append(clean_text)
                    except:
                        clean_texts.append('generic query')  # fallback
                
                self.vectorizer = TfidfVectorizer(
                    max_features=min(1000, len(self.dataset) * 10),
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95
                )
                
                self.query_vectors = self.vectorizer.fit_transform(clean_texts)
                print(f"✅ TF-IDF vectors created: {self.query_vectors.shape}")
                
            except Exception as e:
                print(f"⚠️ TF-IDF creation failed: {str(e)}")
                # Continue without TF-IDF - basic functionality will still work
                self.vectorizer = None
                self.query_vectors = None
            
            print(f"✅ Successfully loaded {len(self.dataset)} query pairs!")
            print(f"📊 Dataset shape: {self.dataset.shape}")
            
            # Show sample of loaded data (safely)
            print("\n📝 Sample loaded data:")
            try:
                for i, row in self.dataset.head(2).iterrows():
                    nl_preview = str(row['natural_language'])[:100] + "..." if len(str(row['natural_language'])) > 100 else str(row['natural_language'])
                    sql_preview = str(row['sql_query'])[:100] + "..." if len(str(row['sql_query'])) > 100 else str(row['sql_query'])
                    print(f"Row {i+1}:")
                    print(f"  NL: {nl_preview}")
                    print(f"  SQL: {sql_preview}")
            except Exception as e:
                print(f"⚠️ Could not display sample data: {str(e)}")
            
            return True
            
        except pd.errors.EmptyDataError:
            print("❌ Error: CSV file is empty")
            return False
        except pd.errors.ParserError as e:
            print(f"❌ Error parsing CSV: {str(e)}")
            print("💡 Try checking if your CSV has proper formatting")
            return False
        except Exception as e:
            print(f"❌ Unexpected error loading dataset: {str(e)}")
            print(f"❌ Error type: {type(e).__name__}")
            import traceback
            print("🔍 Full traceback:")
            traceback.print_exc()
            return False
    
    def find_similar_examples(self, query: str, n_examples: int = 3) -> List[Dict]:
        """Find similar queries from the dataset for few-shot learning"""
        if self.dataset is None:
            print("❌ No dataset loaded")
            return []
            
        if self.vectorizer is None or self.query_vectors is None:
            print("⚠️ TF-IDF not available, returning random examples")
            return self.get_random_examples(n_examples)
        
        try:
            # Clean the input query
            clean_query = str(query).encode('utf-8', errors='ignore').decode('utf-8')
            
            # Transform the input query
            query_vector = self.vectorizer.transform([clean_query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.query_vectors)[0]
            
            # Get top N similar examples
            top_indices = np.argsort(similarities)[-n_examples:][::-1]
            
            examples = []
            for idx in top_indices:
                if similarities[idx] > 0.05:  # Lower threshold for small datasets
                    examples.append({
                        'natural_language': str(self.dataset.iloc[idx]['natural_language']),
                        'sql_query': str(self.dataset.iloc[idx]['sql_query']),
                        'similarity': similarities[idx]
                    })
            
            print(f"🔍 Found {len(examples)} similar examples for query: '{query[:50]}...'")
            return examples
            
        except Exception as e:
            print(f"❌ Error finding similar examples: {str(e)}")
            # Fallback to random examples
            return self.get_random_examples(n_examples)
    
    def get_random_examples(self, n_examples: int = 5) -> List[Dict]:
        """Get random examples from the dataset"""
        if self.dataset is None:
            return []
        
        sample_size = min(n_examples, len(self.dataset))
        sample_df = self.dataset.sample(n=sample_size)
        
        return [
            {
                'natural_language': str(row['natural_language']),
                'sql_query': str(row['sql_query'])
            }
            for _, row in sample_df.iterrows()
        ]
    
    def show_dataset_stats(self):
        """Display dataset statistics"""
        if self.dataset is None:
            print("❌ No dataset loaded")
            return
        
        print(f"📈 Dataset Statistics:")
        print(f"   Total query pairs: {len(self.dataset)}")
        
        try:
            print(f"   Average NL query length: {self.dataset['natural_language'].str.len().mean():.1f} chars")
            print(f"   Average SQL query length: {self.dataset['sql_query'].str.len().mean():.1f} chars")
        except:
            print("   Could not calculate average lengths")
        
        print(f"\n📋 Sample queries:")
        try:
            print(self.dataset.head())
        except:
            print("Could not display sample queries")

class SQLValidator:
    """Basic SQL validation for OMOP queries"""
    
    def __init__(self):
        self.omop_tables = set(OMOPSchemaHandler().omop_tables.keys())
    
    def validate_sql(self, sql_query: str) -> Tuple[bool, List[str]]:
        """Basic validation of generated SQL"""
        warnings = []
        
        # Check for basic SQL structure
        sql_upper = sql_query.upper()
        
        if not sql_upper.startswith('SELECT'):
            warnings.append("Query should start with SELECT")
        
        # Check for OMOP table references
        omop_table_found = False
        for table in self.omop_tables:
            if table.lower() in sql_query.lower():
                omop_table_found = True
                break
        
        if not omop_table_found:
            warnings.append("No OMOP CDM tables found in query")
        
        # Check for common issues
        if 'concept_name' in sql_query.lower() and 'concept' not in sql_query.lower():
            warnings.append("Using concept_name but missing concept table join")
        
        # Check for SQL injection patterns (basic)
        dangerous_patterns = [';--', 'DROP', 'DELETE', 'UPDATE', 'INSERT']
        for pattern in dangerous_patterns:
            if pattern in sql_upper:
                warnings.append(f"Potentially dangerous SQL pattern detected: {pattern}")
        
        is_valid = len(warnings) == 0
        return is_valid, warnings

class LLMProvider(Enum):
    """Supported LLM providers"""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"

class MultiLLMTranslator:
    """Handles translation using multiple LLM providers"""
    
    def __init__(self, provider: LLMProvider, api_key: str, dataset_handler: Optional[DatasetHandler] = None):
        self.provider = provider
        self.api_key = api_key
        self.dataset_handler = dataset_handler
        self.schema_handler = OMOPSchemaHandler()
        self.model = None
        
        # Initialize the appropriate client
        if provider == LLMProvider.ANTHROPIC:
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model_name = "claude-3-5-sonnet-20241022"
        elif provider == LLMProvider.OPENAI:
            self.client = openai.OpenAI(api_key=api_key)
            self.model_name = "gpt-4o"
        elif provider == LLMProvider.GOOGLE:
            genai.configure(api_key=api_key)
            self.model_name = "gemini-1.5-flash"
            self.model = genai.GenerativeModel(self.model_name)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def create_prompt(self, natural_query: str) -> str:
        """Creates a structured prompt for the LLM with few-shot examples"""
        
        # Get similar examples from dataset if available
        examples_text = ""
        if self.dataset_handler and self.dataset_handler.dataset is not None:
            similar_examples = self.dataset_handler.find_similar_examples(natural_query, 3)
            
            if similar_examples:
                examples_text = "\nExamples from expert-written queries:\n"
                for i, example in enumerate(similar_examples, 1):
                    examples_text += f"\nExample {i}:\n"
                    examples_text += f"Natural Language: \"{example['natural_language']}\"\n"
                    examples_text += f"SQL: {example['sql_query']}\n"
        
        prompt = f"""
You are an expert in OMOP Common Data Model (CDM) and SQL. Convert the following natural language query into a valid OMOP CDM SQL query.

{self.schema_handler.get_schema_info()}
{examples_text}

Natural Language Query: "{natural_query}"

Requirements:
1. Generate ONLY the SQL query, no explanations
2. Use proper OMOP CDM table names and column names
3. Include appropriate JOINs when needed
4. Use concept_id for standardized terminology
5. ALWAYS use concept_ancestor table for condition/drug/procedure queries to capture hierarchical relationships
6. Include reasonable date filters when temporal context is mentioned
7. Ensure the query is syntactically correct
8. Use aliases for better readability
9. Follow the patterns shown in the examples above

Example of proper hierarchical querying:
-- Find all patients with diabetes (including all subtypes)
SELECT DISTINCT p.person_id 
FROM person p 
JOIN condition_occurrence co ON p.person_id = co.person_id 
WHERE co.condition_concept_id IN (
  SELECT descendant_concept_id 
  FROM concept_ancestor 
  WHERE ancestor_concept_id = 201820 -- Diabetes mellitus
)

SQL Query:
"""
        return prompt
    
    def translate_query(self, natural_query: str) -> Tuple[str, bool]:
        """Translates natural language to SQL using the selected LLM provider"""
        try:
            prompt = self.create_prompt(natural_query)
            
            if self.provider == LLMProvider.ANTHROPIC:
                return self._translate_with_anthropic(prompt)
            elif self.provider == LLMProvider.OPENAI:
                return self._translate_with_openai(prompt)
            elif self.provider == LLMProvider.GOOGLE:
                return self._translate_with_google(prompt)
            else:
                return "Unsupported provider", False
                
        except Exception as e:
            return f"Error generating SQL with {self.provider.value}: {str(e)}", False
    
    def _translate_with_anthropic(self, prompt: str) -> Tuple[str, bool]:
        """Translate using Anthropic Claude"""
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            sql_query = response.content[0].text.strip()
            sql_query = self._clean_sql_response(sql_query)
            return sql_query, True
            
        except Exception as e:
            return f"Anthropic API error: {str(e)}", False
    
    def _translate_with_openai(self, prompt: str) -> Tuple[str, bool]:
        """Translate using OpenAI GPT"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert OMOP CDM SQL developer. Generate only SQL queries without explanations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            sql_query = response.choices[0].message.content.strip()
            sql_query = self._clean_sql_response(sql_query)
            return sql_query, True
            
        except Exception as e:
            return f"OpenAI API error: {str(e)}", False
    
    def _translate_with_google(self, prompt: str) -> Tuple[str, bool]:
        """Translate using Google Gemini"""
        try:
            response = self.model.generate_content(prompt)
            sql_query = response.text.strip()
            sql_query = self._clean_sql_response(sql_query)
            return sql_query, True
            
        except Exception as e:
            return f"Google API error: {str(e)}", False
    
    def _clean_sql_response(self, sql_query: str) -> str:
        """Clean up the SQL response by removing markdown formatting"""
        # Remove markdown code blocks
        sql_query = re.sub(r'```sql\n?', '', sql_query)
        sql_query = re.sub(r'```\n?', '', sql_query)
        sql_query = sql_query.strip()
        
        # Remove any explanatory text before/after the SQL
        lines = sql_query.split('\n')
        sql_lines = []
        in_sql = False
        
        for line in lines:
            line = line.strip()
            if line.upper().startswith('SELECT') or line.upper().startswith('WITH'):
                in_sql = True
            if in_sql:
                sql_lines.append(line)
            # Stop if we hit explanatory text after SQL
            if in_sql and line and not any(keyword in line.upper() for keyword in 
                ['SELECT', 'FROM', 'WHERE', 'JOIN', 'AND', 'OR', 'GROUP', 'ORDER', 'HAVING', 'LIMIT', 'WITH', '--']):
                break
        
        return '\n'.join(sql_lines).strip() 
        
# Updated main function section for LLM provider selection
def get_llm_provider_ui():
    """Create the LLM provider selection UI"""
    
    st.markdown("### 🤖 AI Provider")
    
    # Provider selection
    provider_options = {
        "🔥 Anthropic Claude (Recommended)": LLMProvider.ANTHROPIC,
        "🚀 OpenAI GPT-4": LLMProvider.OPENAI,
        "🌟 Google Gemini": LLMProvider.GOOGLE
    }
    
    selected_provider = st.selectbox(
        "Choose AI Provider:",
        list(provider_options.keys()),
        help="Claude Sonnet performs best for complex SQL generation"
    )
    
    provider = provider_options[selected_provider]
    
    # Provider-specific information
    if provider == LLMProvider.ANTHROPIC:
        st.info("💡 **Claude Sonnet** excels at complex SQL and OMOP queries")
        api_help = "Get your Claude API key from [Anthropic Console](https://console.anthropic.com/)"
        placeholder = "sk-ant-..."
    elif provider == LLMProvider.OPENAI:
        st.info("💡 **GPT-4** offers reliable SQL generation with good reasoning")
        api_help = "Get your OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)"
        placeholder = "sk-..."
    else:  # Google
        st.info("💡 **Gemini** provides fast, free SQL generation")
        api_help = "Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)"
        placeholder = "AIza..."
    
    # API Key input
    api_key = st.text_input(
        "Enter your API key:",
        type="password",
        help=api_help,
        placeholder=placeholder
    )
    
    # Show pricing info
    with st.expander("💰 Pricing Information"):
        if provider == LLMProvider.ANTHROPIC:
            st.markdown("""
            **Claude Sonnet 3.5:**
            - Input: $3 per 1M tokens
            - Output: $15 per 1M tokens
            - ~1000 queries ≈ $0.50
            """)
        elif provider == LLMProvider.OPENAI:
            st.markdown("""
            **GPT-4o:**
            - Input: $2.50 per 1M tokens  
            - Output: $10 per 1M tokens
            - ~1000 queries ≈ $0.40
            """)
        else:
            st.markdown("""
            **Gemini 1.5 Pro:**
            - 15 requests/minute free
            - 1500 requests/day free
            - Perfect for demos!
            """)
    
    return provider, api_key

def main():
    """Main Streamlit application optimized for Streamlit Cloud"""
    st.set_page_config(
        page_title="OMOP NLP-to-SQL Agent",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header with GitHub link
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🏥 OMOP NLP-to-SQL Agent")
        st.markdown("Convert natural language queries into OMOP Common Data Model SQL queries")
    
    with col3:
        st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/AlekseiZakh/omop-nlp-sql-agent)")
    
    # Initialize dataset handler
    dataset_handler = DatasetHandler()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # LLM Provider Selection
        provider, api_key = get_llm_provider_ui()
        
        st.divider()
        
        # Dataset upload section
        st.header("📊 Your Dataset (Optional)")
        st.markdown("Upload your custom NL-to-SQL pairs to improve accuracy")
        
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="CSV with columns: natural_language, sql_query (or similar names)"
        )
        
        if uploaded_file is not None:
            with st.spinner("🔄 Loading dataset..."):
                try:
                    # Debug info
                    st.write(f"📁 File name: {uploaded_file.name}")
                    st.write(f"📊 File size: {uploaded_file.size} bytes")
                    
                    # Reset file pointer to beginning
                    uploaded_file.seek(0)
                    
                    # Try to load the dataset directly with the uploaded file
                    if dataset_handler.load_dataset(uploaded_file):
                        st.success(f"✅ Loaded {len(dataset_handler.dataset)} query pairs!")
                        
                        # Show dataset stats
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Query Pairs", len(dataset_handler.dataset))
                        with col2:
                            avg_length = dataset_handler.dataset['natural_language'].str.len().mean()
                            st.metric("Avg NL Length", f"{avg_length:.0f} chars")
                        
                        with st.expander("📋 Dataset Preview"):
                            st.dataframe(
                                dataset_handler.dataset.head(3),
                                use_container_width=True
                            )
                            
                        # Show column detection info
                        st.info(f"📋 Detected columns: '{uploaded_file.name}' appears to have natural language and SQL columns")
                        
                    else:
                        st.error("❌ Failed to load dataset")
                        
                        # Provide debugging help
                        st.write("🔍 **Debugging information:**")
                        
                        # Try to read and show first few lines
                        try:
                            uploaded_file.seek(0)
                            content = uploaded_file.read().decode('utf-8')
                            lines = content.split('\n')[:5]
                            
                            st.write("📝 **First 5 lines of your file:**")
                            for i, line in enumerate(lines):
                                if line.strip():
                                    st.code(f"Line {i+1}: {line}")
                                    
                        except Exception as e:
                            st.write(f"❌ Could not read file: {str(e)}")
                        
                        # Show expected format
                        st.write("📋 **Expected CSV format:**")
                        example_data = {
                            'natural_language': ['Find patients with diabetes', 'Show hypertension cases'],
                            'sql_query': ['SELECT * FROM...', 'SELECT * FROM...']
                        }
                        st.dataframe(pd.DataFrame(example_data))
                        
                        st.write("""
                        **💡 Common issues:**
                        - Column names should contain keywords like: question, query, natural, sql, answer
                        - File should be properly formatted CSV
                        - No empty rows at the beginning
                        - UTF-8 encoding recommended
                        """)
                        
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
                    st.write("🔍 **Error details:**")
                    st.code(str(e))
                    
                    # Show troubleshooting tips
                    with st.expander("🛠️ Troubleshooting Tips"):
                        st.markdown("""
                        **Try these solutions:**
                        
                        1. **Check file format:**
                           - Save as CSV (UTF-8)
                           - Remove any special characters
                           - Ensure no empty rows at top
                        
                        2. **Check column names:**
                           - Use simple names like: `question`, `sql`
                           - Or: `natural_language`, `sql_query`
                           - Avoid spaces or special characters
                        
                        3. **Check data:**
                           - At least 2 columns required
                           - No completely empty rows
                           - Text should be properly quoted if contains commas
                        
                        4. **Test with sample:**
                           Try uploading a simple 2-row CSV first
                        """)
                        
                        # Provide a downloadable sample
                        sample_csv = """natural_language,sql_query
"Find patients with diabetes","SELECT DISTINCT p.person_id FROM person p JOIN condition_occurrence co ON p.person_id = co.person_id"
"Show hypertension cases","SELECT * FROM condition_occurrence WHERE condition_concept_id = 316866"
"""
                        st.download_button(
                            label="📥 Download Sample CSV Format",
                            data=sample_csv,
                            file_name="sample_omop_queries.csv",
                            mime="text/csv"
                        )
        
        # About section
        st.divider()
        st.markdown("### 👨‍⚕️ About")
        st.markdown("""
        Created by Aleksey Zakharov Medical Doctor & Data Scientist
        
        **Features:**
        - Multiple AI providers (Claude, GPT-4, Gemini)
        - Expert OMOP CDM knowledge
        - Hierarchical concept relationships
        - Few-shot learning with your data
        - SQL validation & safety checks
        """)
    
    # Main interface
    if api_key:
        try:
            # Initialize the multi-LLM translator
            translator = MultiLLMTranslator(provider, api_key, dataset_handler)
            validator = SQLValidator()
            
            # Show which provider is being used
            provider_names = {
                LLMProvider.ANTHROPIC: "🔥 Claude Sonnet 3.5",
                LLMProvider.OPENAI: "🚀 GPT-4o", 
                LLMProvider.GOOGLE: "🌟 Gemini 1.5 Pro"
            }
            
            st.success(f"✅ Connected to {provider_names[provider]}")
            
            # Show examples section
            if dataset_handler.dataset is not None:
                st.subheader("📝 Examples from your dataset:")
                examples = dataset_handler.get_random_examples(4)
                
                cols = st.columns(len(examples))
                for i, example in enumerate(examples):
                    with cols[i]:
                        # Truncate long queries for button display
                        display_text = example['natural_language']
                        if len(display_text) > 50:
                            display_text = display_text[:47] + "..."
                        
                        if st.button(
                            f"📋 Example {i+1}", 
                            key=f"dataset_example_{i}",
                            help=example['natural_language'],
                            use_container_width=True
                        ):
                            st.session_state.query_input = example['natural_language']
            else:
                # Default examples
                st.subheader("📝 Try these example queries:")
                example_queries = [
                    "Find all patients with diabetes diagnosed in the last 2 years",
                    "Show patients over 65 with hypertension", 
                    "List patients who had a heart attack and their current medications",
                    "Count patients by gender and race"
                ]
                
                cols = st.columns(len(example_queries))
                for i, example in enumerate(example_queries):
                    with cols[i]:
                        if st.button(
                            f"📋 Example {i+1}", 
                            key=f"example_{i}",
                            help=example,
                            use_container_width=True
                        ):
                            st.session_state.query_input = example
            
            st.divider()
            
            # Query input section
            st.subheader("💬 Enter Your Query")
            query_input = st.text_area(
                "Natural language query:",
                value=st.session_state.get('query_input', ''),
                height=120,
                placeholder="e.g., Find all patients with diabetes who are over 65 years old and taking metformin",
                help="Describe what you want to find in plain English"
            )
            
            # Translate button
            if st.button("🔄 Translate to SQL", type="primary", use_container_width=True):
                if not query_input.strip():
                    st.error("Please enter a query first!")
                else:
                    # Show which AI is being used
                    ai_names = {
                        LLMProvider.ANTHROPIC: "Claude Sonnet 3.5",
                        LLMProvider.OPENAI: "GPT-4o",
                        LLMProvider.GOOGLE: "Gemini 1.5 Pro"
                    }
                    
                    with st.spinner(f"🧠 {ai_names[provider]} is translating your query..."):
                        # Show similar examples if dataset is loaded
                        if dataset_handler.dataset is not None:
                            similar_examples = dataset_handler.find_similar_examples(query_input, 3)
                            if similar_examples:
                                with st.expander("🔍 Similar examples from your dataset (used for context)"):
                                    for i, example in enumerate(similar_examples, 1):
                                        similarity_color = "🟢" if example['similarity'] > 0.5 else "🟡" if example['similarity'] > 0.3 else "🔴"
                                        st.write(f"{similarity_color} **Example {i}** (similarity: {example['similarity']:.2f})")
                                        st.write(f"*Query:* {example['natural_language']}")
                                        st.code(example['sql_query'], language="sql")
                                        if i < len(similar_examples):
                                            st.divider()
                        
                        # Generate SQL
                        sql_query, success = translator.translate_query(query_input)
                        
                        if success:
                            # Validate the generated SQL
                            is_valid, warnings = validator.validate_sql(sql_query)
                            
                            st.subheader(f"🎯 Generated SQL Query (via {ai_names[provider]}):")
                            st.code(sql_query, language="sql")
                            
                            # Show validation results
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                if is_valid:
                                    st.success("✅ SQL query validation passed!")
                                else:
                                    st.warning("⚠️ Potential issues detected:")
                                    for warning in warnings:
                                        st.write(f"• {warning}")
                            
                            with col2:
                                # Copy button (visual feedback)
                                if st.button("📋 Copy SQL", help="Click to select and copy"):
                                    st.success("✅ Ready to copy!")
                            
                            # Additional query info
                            with st.expander("📊 Query Analysis"):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("SQL Length", f"{len(sql_query)} chars")
                                with col2:
                                    table_count = len([table for table in ['person', 'condition_occurrence', 'drug_exposure', 'procedure_occurrence', 'measurement'] if table in sql_query.lower()])
                                    st.metric("Tables Used", table_count)
                                with col3:
                                    has_hierarchy = "concept_ancestor" in sql_query.lower()
                                    st.metric("Uses Hierarchy", "✅" if has_hierarchy else "❌")
                                
                                if has_hierarchy:
                                    st.info("🎯 Great! This query uses concept_ancestor for proper hierarchical relationships.")
                                else:
                                    st.warning("💡 Consider if this query should use concept_ancestor for more comprehensive results.")
                            
                        else:
                            st.error(f"❌ Translation failed: {sql_query}")
            
            # Evaluation section if dataset is loaded
            if dataset_handler.dataset is not None:
                st.divider()
                with st.expander("🧪 Evaluate Model Performance"):
                    st.markdown("Test the model against your expert-written queries")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        n_samples = st.slider("Number of test samples:", 1, 10, 3)
                    with col2:
                        if st.button("🚀 Run Evaluation"):
                            sample_queries = dataset_handler.get_random_examples(n_samples)
                            
                            for i, example in enumerate(sample_queries, 1):
                                st.write(f"**Test Case {i}:**")
                                st.write(f"*Input:* {example['natural_language']}")
                                
                                col_expert, col_generated = st.columns(2)
                                
                                with col_expert:
                                    st.write("**Your Expert SQL:**")
                                    st.code(example['sql_query'], language="sql")
                                
                                with col_generated:
                                    st.write("**Generated SQL:**")
                                    generated_sql, success = translator.translate_query(example['natural_language'])
                                    if success:
                                        st.code(generated_sql, language="sql")
                                        
                                        # Simple similarity check
                                        similarity = len(set(generated_sql.lower().split()) & set(example['sql_query'].lower().split())) / max(len(set(generated_sql.lower().split())), len(set(example['sql_query'].lower().split())))
                                        
                                        if similarity > 0.6:
                                            st.success(f"🎯 Good match! ({similarity:.1%} similarity)")
                                        elif similarity > 0.3:
                                            st.warning(f"⚠️ Partial match ({similarity:.1%} similarity)")
                                        else:
                                            st.error(f"❌ Low match ({similarity:.1%} similarity)")
                                    else:
                                        st.error("Failed to generate")
                                
                                if i < len(sample_queries):
                                    st.divider()
            
            # Schema reference
            with st.expander("📚 OMOP CDM Schema Reference"):
                st.markdown("### Core Tables Overview")
                schema_handler = OMOPSchemaHandler()
                
                # Create tabs for better organization
                tab_names = list(schema_handler.omop_tables.keys())
                tabs = st.tabs([t.replace('_', ' ').title() for t in tab_names])
                
                for tab, table_name in zip(tabs, tab_names):
                    with tab:
                        table_info = schema_handler.omop_tables[table_name]
                        st.write(f"**{table_info['description']}**")
                        st.write(f"**Primary Key:** `{table_info['primary_key']}`")
                        st.write("**Columns:**")
                        for col in table_info['columns']:
                            st.write(f"• `{col}`")
            
        except Exception as e:
            st.error(f"❌ Failed to initialize {provider.value}: {str(e)}")
            st.info("💡 Please check your API key and try again")
    
    else:
        # Landing page when no API key
        st.markdown("""
        ## 🎯 What This Tool Does
        
        Transform natural language questions about healthcare data into precise OMOP Common Data Model SQL queries.
        
        **Perfect for:**
        - Healthcare researchers
        - Clinical data analysts  
        - OMOP CDM users
        - Medical informaticians
        """)
        
        # Demo section
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Input Example")
            st.info("Find all patients with diabetes who are over 65 years old and taking metformin")
        
        with col2:
            st.subheader("🎯 Generated OMOP SQL")
            demo_sql = """SELECT DISTINCT p.person_id, p.year_of_birth
FROM person p 
JOIN condition_occurrence co ON p.person_id = co.person_id 
JOIN drug_exposure de ON p.person_id = de.person_id
WHERE co.condition_concept_id IN (
  SELECT descendant_concept_id 
  FROM concept_ancestor 
  WHERE ancestor_concept_id = 201820 -- Diabetes
)
AND (2024 - p.year_of_birth) > 65
AND de.drug_concept_id IN (
  SELECT descendant_concept_id 
  FROM concept_ancestor 
  WHERE ancestor_concept_id = 1503297 -- Metformin
)"""
            st.code(demo_sql, language="sql")
        
        st.markdown("### ✨ Key Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🧠 Smart Translation**
            - Multiple AI providers
            - Understands medical terminology
            - Handles complex queries
            """)
        
        with col2:
            st.markdown("""
            **🏥 OMOP Expert**
            - Proper concept hierarchies
            - Uses concept_ancestor
            - Follows best practices
            """)
        
        with col3:
            st.markdown("""
            **📊 Your Data**
            - Upload custom examples
            - Few-shot learning
            - Performance evaluation
            """)
        
        st.info("👈 **Get started:** Choose an AI provider and enter your API key in the sidebar!")

if __name__ == "__main__":
    main()
