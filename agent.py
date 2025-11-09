from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.llms import Ollama
from langchain.agents.agent_types import AgentType
import pandas as pd
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

class UniversalDataAgent:
    """Enhanced AI agent for multi-domain data analysis"""
    
    def __init__(self, use_ollama=True, api_key=None):
        """Initialize agent with local or cloud LLM"""
        self.use_ollama = use_ollama
        
        if use_ollama:
            print("Initializing Ollama LLM...")
            self.llm = Ollama(
                model="llama3.2:3b",
                temperature=0
            )
        else:
            from langchain_google_genai import ChatGoogleGenerativeAI
            print("Initializing Gemini LLM...")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=api_key,
                temperature=0
            )
        
        self.agent = None
        self.df = None
        self.file_type = None
        self.detected_domain = None
    
    def load_data(self, file_path):
        """Load CSV or Excel files automatically"""
        file_extension = Path(file_path).suffix.lower()
        
        print(f" Loading file: {Path(file_path).name}")
        
        try:
            if file_extension == '.csv':
                self.df = pd.read_csv(file_path)
                self.file_type = 'CSV'
            elif file_extension in ['.xlsx', '.xlsm']:
                self.df = pd.read_excel(file_path, engine='openpyxl')
                self.file_type = 'Excel (XLSX)'
            elif file_extension == '.xls':
                self.df = pd.read_excel(file_path, engine='xlrd')
                self.file_type = 'Excel (XLS)'
            elif file_extension == '.tsv':
                self.df = pd.read_csv(file_path, sep='\t')
                self.file_type = 'TSV'
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            print(f"Loaded {self.file_type} with {len(self.df)} rows, {len(self.df.columns)} columns")
            
            # Auto-detect domain
            self.auto_detect_domain()
            
            return self.df.head(), self.df.shape, self.df.columns.tolist(), self.file_type
            
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")
    
    def get_excel_sheets(self, file_path):
        """Get list of sheets in Excel file"""
        excel_file = pd.ExcelFile(file_path)
        return excel_file.sheet_names
    
    def load_excel_sheet(self, file_path, sheet_name):
        """Load specific sheet from Excel file"""
        self.df = pd.read_excel(file_path, sheet_name=sheet_name)
        self.file_type = f'Excel (Sheet: {sheet_name})'
        self.auto_detect_domain()
        return self.df.head(), self.df.shape, self.df.columns.tolist()
    
    def auto_detect_domain(self):
        """Automatically detect data domain from column names"""
        if self.df is None:
            return None
        
        cols_lower = [col.lower() for col in self.df.columns]
        
        # Domain detection rules
        domains = {
            'Healthcare/Medical': ['patient', 'diagnosis', 'treatment', 'bmi', 'insurance', 'charges', 'smoker', 'medical'],
            'Sales/E-commerce': ['revenue', 'sales', 'profit', 'customer', 'product', 'quantity', 'order', 'discount', 'ship'],
            'Finance': ['price', 'stock', 'investment', 'portfolio', 'ticker', 'market', 'return'],
            'HR/Recruitment': ['salary', 'employee', 'department', 'position', 'hire', 'job', 'tenure'],
            'Real Estate': ['property', 'bedrooms', 'bathrooms', 'sqft', 'location', 'area', 'rooms'],
            'Transportation': ['passenger', 'survived', 'embarked', 'ticket', 'cabin', 'pclass', 'fare'],
            'Telecommunications': ['churn', 'contract', 'tenure', 'monthlycharges', 'phoneservice', 'internetservice'],
            'Social/Happiness': ['happiness', 'gdp', 'freedom', 'corruption', 'generosity', 'country'],
            'Weather/Climate': ['temperature', 'humidity', 'precipitation', 'wind', 'pressure'],
            'Neuroimaging': ['subject', 'fmri', 'scan', 'roi', 'activation', 'motion', 'task']
        }
        
        matches = {}
        for domain, keywords in domains.items():
            match_count = sum(1 for kw in keywords if any(kw in col for col in cols_lower))
            if match_count > 0:
                matches[domain] = match_count
        
        if matches:
            self.detected_domain = max(matches, key=matches.get)
        else:
            self.detected_domain = 'General'
        
        print(f"Detected domain: {self.detected_domain}")
        return self.detected_domain
    
    def create_agent(self):
        """Create pandas dataframe agent"""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print("Creating AI agent...")
        self.agent = create_pandas_dataframe_agent(
            self.llm,
            self.df,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            allow_dangerous_code=True,
            handle_parsing_errors=True,
            max_iterations=10
        )
        print("Agent ready!")
        return self.agent
    
    def query(self, question):
        """Ask natural language question"""
        if self.agent is None:
            self.create_agent()
        
        try:
            print(f"\nQuestion: {question}")
            response = self.agent.invoke(question)
            return response['output']
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_domain_insights(self):
        """Get domain-specific insights"""
        domain = self.detected_domain or 'General'
        
        insights = {
            'domain': domain,
            'suggestions': [],
            'key_metrics': [],
            'example_questions': []
        }
        
        if domain == 'Healthcare/Medical':
            insights['suggestions'] = [
                "Analyze patient demographics by age and gender",
                "Calculate average medical charges by region",
                "Compare costs between smokers and non-smokers"
            ]
            insights['key_metrics'] = ['Average charges', 'Patient count', 'BMI distribution']
            insights['example_questions'] = [
                "What is the average insurance charge?",
                "How many smokers vs non-smokers?",
                "Correlation between BMI and charges"
            ]
        
        elif domain == 'Sales/E-commerce':
            insights['suggestions'] = [
                "Calculate total revenue by product category",
                "Identify top-performing regions",
                "Analyze profit margins and discount impact"
            ]
            insights['key_metrics'] = ['Total revenue', 'Average profit', 'Top products']
            insights['example_questions'] = [
                "What is the total sales?",
                "Which category has highest profit?",
                "Average discount by region"
            ]
        
        elif domain == 'Transportation':
            insights['suggestions'] = [
                "Calculate survival rate by passenger class",
                "Analyze age distribution of survivors",
                "Compare fares across embarkation ports"
            ]
            insights['key_metrics'] = ['Survival rate', 'Average age', 'Fare distribution']
            insights['example_questions'] = [
                "What percentage survived?",
                "Average age by passenger class",
                "Survival rate for males vs females"
            ]
        
        elif domain == 'Telecommunications':
            insights['suggestions'] = [
                "Calculate churn rate by contract type",
                "Analyze monthly charges distribution",
                "Compare tenure for churned vs retained customers"
            ]
            insights['key_metrics'] = ['Churn rate', 'Average tenure', 'Revenue impact']
            insights['example_questions'] = [
                "What is the overall churn rate?",
                "Average monthly charges by contract?",
                "How many customers have fiber optic?"
            ]
        
        elif domain == 'Social/Happiness':
            insights['suggestions'] = [
                "Rank countries by happiness score",
                "Analyze correlation between GDP and happiness",
                "Compare regional happiness trends"
            ]
            insights['key_metrics'] = ['Average happiness', 'Top countries', 'GDP correlation']
            insights['example_questions'] = [
                "Which country has highest happiness?",
                "Correlation between GDP and happiness",
                "Average life expectancy by region"
            ]
        
        elif domain == 'Real Estate':
            insights['suggestions'] = [
                "Calculate average price by number of bedrooms",
                "Analyze price per square foot trends",
                "Compare property values by location"
            ]
            insights['key_metrics'] = ['Median price', 'Price per sqft', 'Property count']
            insights['example_questions'] = [
                "What is the average property price?",
                "Price difference by location",
                "How many properties have 3+ bedrooms?"
            ]
        
        elif domain == 'Neuroimaging':
            insights['suggestions'] = [
                "Analyze subject demographics",
                "Compare groups on behavioral measures",
                "Check data quality and missing values"
            ]
            insights['key_metrics'] = ['Sample size', 'Group differences', 'Data completeness']
            insights['example_questions'] = [
                "What is the average age?",
                "How many subjects per group?",
                "Any missing fMRI data?"
            ]
        
        else:
            insights['suggestions'] = [
                "Explore basic statistics",
                "Check for missing values",
                "Analyze distributions"
            ]
            insights['key_metrics'] = ['Row count', 'Column types', 'Missing data']
            insights['example_questions'] = [
                "How many rows and columns?",
                "Are there missing values?",
                "Show summary statistics"
            ]
        
        return insights
    
    def generate_smart_questions(self):
        """Generate relevant questions based on data structure"""
        if self.df is None:
            return []
        
        questions = []
        
        # Basic questions
        questions.append(f"How many rows are in this dataset?")
        
        # Check for missing values
        if self.df.isnull().sum().sum() > 0:
            questions.append("Which columns have missing values?")
        
        # Numeric columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            questions.append(f"What is the average {numeric_cols[0]}?")
            if len(numeric_cols) > 1:
                questions.append(f"Show correlation between {numeric_cols[0]} and {numeric_cols[1]}")
        
        # Categorical columns
        cat_cols = self.df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            questions.append(f"Show distribution of {cat_cols[0]}")
        
        # Domain-specific questions
        insights = self.get_domain_insights()
        questions.extend(insights['example_questions'][:3])
        
        return questions[:8]
