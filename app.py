import streamlit as st
import google.generativeai as genai
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
    
    def load_dataset(self, csv_path: str) -> bool:
        """Load the CSV dataset with natural language and SQL pairs"""
        try:
            # Try different possible column names
            df = pd.read_csv(csv_path)
            
            # Auto-detect column names (flexible naming)
            possible_nl_cols = ['natural_language', 'nl_query', 'question', 'query', 'natural_query']
            possible_sql_cols = ['sql', 'sql_query', 'omop_sql', 'answer', 'sql_answer']
            
            nl_col = None
            sql_col = None
            
            for col in df.columns:
                col_lower = col.lower().strip()
                if col_lower in possible_nl_cols:
                    nl_col = col
                elif col_lower in possible_sql_cols:
                    sql_col = col
            
            if nl_col is None or sql_col is None:
                # If auto-detection fails, use first two columns
                columns = df.columns.tolist()
                nl_col = columns[0]
                sql_col = columns[1]
                print(f"⚠️ Auto-detection failed, using columns: {nl_col}, {sql_col}")
            
            self.dataset = df[[nl_col, sql_col]].copy()
            self.dataset.columns = ['natural_language', 'sql_query']
            
            # Remove any empty rows
            self.dataset = self.dataset.dropna().reset_index(drop=True)
            
            # Create TF-IDF vectors for similarity matching
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            self.query_vectors = self.vectorizer.fit_transform(
                self.dataset['natural_language']
            )
            
            print(f"✅ Successfully loaded {len(self.dataset)} query pairs!")
            print(f"📊 Dataset shape: {self.dataset.shape}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
            return False
    
    def find_similar_examples(self, query: str, n_examples: int = 3) -> List[Dict]:
        """Find similar queries from the dataset for few-shot learning"""
        if self.dataset is None or self.vectorizer is None:
            return []
        
        try:
            # Transform the input query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.query_vectors)[0]
            
            # Get top N similar examples
            top_indices = np.argsort(similarities)[-n_examples:][::-1]
            
            examples = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Only include reasonably similar examples
                    examples.append({
                        'natural_language': self.dataset.iloc[idx]['natural_language'],
                        'sql_query': self.dataset.iloc[idx]['sql_query'],
                        'similarity': similarities[idx]
                    })
            
            return examples
            
        except Exception as e:
            print(f"❌ Error finding similar examples: {str(e)}")
            return []
    
    def get_random_examples(self, n_examples: int = 5) -> List[Dict]:
        """Get random examples from the dataset"""
        if self.dataset is None:
            return []
        
        sample_size = min(n_examples, len(self.dataset))
        sample_df = self.dataset.sample(n=sample_size)
        
        return [
            {
                'natural_language': row['natural_language'],
                'sql_query': row['sql_query']
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
        print(f"   Average NL query length: {self.dataset['natural_language'].str.len().mean():.1f} chars")
        print(f"   Average SQL query length: {self.dataset['sql_query'].str.len().mean():.1f} chars")
        print(f"\n📋 Sample queries:")
        print(self.dataset.head())

print("✅ Dataset Handler class defined!")

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

class GoogleGeminiTranslator:
    """Handles translation from natural language to OMOP SQL using Google Gemini"""
    
    def __init__(self, api_key: str, dataset_handler: Optional[DatasetHandler] = None):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.schema_handler = OMOPSchemaHandler()
        self.dataset_handler = dataset_handler
    
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
        """Translates natural language to SQL using Gemini"""
        try:
            prompt = self.create_prompt(natural_query)
            response = self.model.generate_content(prompt)
            
            # Extract SQL from response
            sql_query = response.text.strip()
            
            # Clean up the response (remove markdown formatting if present)
            sql_query = re.sub(r'```sql\n?', '', sql_query)
            sql_query = re.sub(r'```\n?', '', sql_query)
            sql_query = sql_query.strip()
            
            return sql_query, True
            
        except Exception as e:
            return f"Error generating SQL: {str(e)}", False

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
        
        # API Key input with better instructions
        st.markdown("### 🔑 Google Gemini API Key")
        api_key = st.text_input(
            "Enter your API key:",
            type="password",
            help="Get your free API key from Google AI Studio"
        )
        
        if not api_key:
            st.info("👆 Enter your API key to get started")
            with st.expander("📝 How to get a free API key"):
                st.markdown("""
                1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
                2. Sign in with your Google account
                3. Click "Create API Key"
                4. Copy and paste it above
                
                **Free tier includes:**
                - 15 requests per minute
                - Perfect for portfolio demos!
                """)
        
        st.divider()
        
        # Dataset upload section
        st.header("📊 Your Dataset (Optional)")
        st.markdown("Upload your custom NL-to-SQL pairs to improve accuracy")
        
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="CSV with columns: natural_language, sql_query"
        )
        
        if uploaded_file is not None:
            with st.spinner("🔄 Loading dataset..."):
                # Create a temporary file path for the uploaded file
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                if dataset_handler.load_dataset(tmp_path):
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
                else:
                    st.error("❌ Failed to load dataset")
                
                # Clean up temp file
                os.unlink(tmp_path)
        
        # About section
        st.divider()
        st.markdown("### 👨‍⚕️ About")
        st.markdown("""
        Created by a Medical Doctor & Data Scientist
        
        **Features:**
        - Expert OMOP CDM knowledge
        - Hierarchical concept relationships
        - Few-shot learning with your data
        - SQL validation & safety checks
        """)
    
    # Main interface
    if api_key:
        translator = GoogleGeminiTranslator(api_key, dataset_handler)
        validator = SQLValidator()
        
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
                with st.spinner("🧠 Translating your query..."):
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
                        
                        st.subheader("🎯 Generated SQL Query:")
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
            - Uses Google Gemini AI
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
        
        st.info("👈 **Get started:** Enter your Google Gemini API key in the sidebar!")

if __name__ == "__main__":
    main()
