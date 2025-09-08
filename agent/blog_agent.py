#!/usr/bin/env python3
"""
Blog Generation AI Agent - Complete Single File Implementation
Run with: python blog_agent.py
"""


# this is the complete single file implementation of the blog generation agent
import os
import json
import logging
import asyncio
from datetime import datetime, timedelta , UTC

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import uuid

# Third-party imports
try:
    import google.generativeai as genai
    from tavily import TavilyClient
    from pymongo import MongoClient
    from langgraph.graph import StateGraph, END
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install langgraph google-generativeai tavily-python pymongo python-dotenv")
    exit(1)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('blog_agent.log')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION AND SETTINGS
# ============================================================================

class Settings:
    """Application settings from environment variables"""
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
    MONGODB_URL = os.getenv("MONGODB_URI", "mongodb://localhost:27017/blog_agent")
    
    @classmethod
    def validate(cls):
        """Validate required environment variables"""
        if not cls.GEMINI_API_KEY:
            print("❌ GEMINI_API_KEY not found in environment")
            print("Get your API key from: https://makersuite.google.com/app/apikey")
            return False
            
        if not cls.TAVILY_API_KEY:
            print("❌ TAVILY_API_KEY not found in environment")
            print("Get your API key from: https://tavily.com")
            return False
            
        print(" Environment variables validated")
        return True

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class BlogContent:
    """Blog content structure"""
    title: str
    content: str
    meta_description: str
    tags: List[str]
    seo_keywords: List[str]
    word_count: int

@dataclass
class ConversationMemory:
    """Conversation memory structure"""
    user_id: str
    thread_id: str
    query: str
    response: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        """Convert to dictionary for MongoDB storage"""
        return {
            'user_id': self.user_id,
            'thread_id': self.thread_id,
            'query': self.query,
            'response': self.response,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }

@dataclass
class AgentState:
    """Agent state for LangGraph workflow"""
    query: str
    user_id: str
    thread_id: str
    is_blog_request: bool = False
    search_results: List[Dict[str, Any]] = field(default_factory=list)
    blog_content: Optional[BlogContent] = None
    response: str = ""
    memory_context: List[str] = field(default_factory=list)
    error: Optional[str] = None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_current_user() -> str:
    """Get current user ID (demo implementation)"""
    return "demo_user_123"

def validate_user_access(user_id: str, thread_id: str) -> bool:
    """Validate user access to thread"""
    return True  # Demo implementation

# ============================================================================
# MEMORY MANAGER
# ============================================================================

class MemoryManager:
    """MongoDB-based memory management"""
    
    def __init__(self):
        try:
            self.client = MongoClient(Settings.MONGODB_URL)
            self.db = self.client.blog_agent
            self.conversations = self.db.conversations
            
            # Create indexes for better performance
            self.conversations.create_index([("user_id", 1), ("thread_id", 1)])
            self.conversations.create_index([("timestamp", -1)])
            
            # Test connection
            self.client.server_info()
            logger.info(" MongoDB connected successfully")
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            logger.info("Using in-memory storage as fallback")
            self.conversations = None
            self._memory_store = []

    def save_conversation(self, memory: ConversationMemory) -> str:
        """Save conversation to storage"""
        try:
            if self.conversations is not None:
                result = self.conversations.insert_one(memory.to_dict())
                logger.info(f"Conversation saved with ID: {result.inserted_id}")
                return str(result.inserted_id)
            else:
                # Fallback to in-memory storage
                memory_dict = memory.to_dict()
                memory_dict['_id'] = str(uuid.uuid4())
                self._memory_store.append(memory_dict)
                logger.info("Conversation saved to memory store")
                return memory_dict['_id']
                
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            return ""

    def get_thread_history(self, user_id: str, thread_id: str, limit: int = 10) -> List[ConversationMemory]:
        """Get conversation history for a thread"""
        try:
            if self.conversations is not None:
                cursor = self.conversations.find(
                    {"user_id": user_id, "thread_id": thread_id}
                ).sort("timestamp", -1).limit(limit)
                
                return [ConversationMemory(**doc) for doc in cursor]
            else:
                # Fallback to in-memory storage
                filtered = [m for m in self._memory_store 
                           if m['user_id'] == user_id and m['thread_id'] == thread_id]
                filtered.sort(key=lambda x: x['timestamp'], reverse=True)
                return [ConversationMemory(**m) for m in filtered[:limit]]
                
        except Exception as e:
            logger.error(f"Error retrieving thread history: {e}")
            return []

    def get_user_context(self, user_id: str, hours: int = 24) -> List[str]:
        """Get recent user context"""
        try:
            cutoff_time = datetime.now(UTC) - timedelta(hours=hours)
            
            if self.conversations is not None:
                cursor = self.conversations.find(
                    {
                        "user_id": user_id,
                        "timestamp": {"$gte": cutoff_time}
                    }
                ).sort("timestamp", -1).limit(20)
                
                context = []
                for doc in cursor:
                    context.append(f"User: {doc['query']}")
                    context.append(f"Assistant: {doc['response'][:200]}...")
                    
                return context
            else:
                # Fallback to in-memory storage
                filtered = [m for m in self._memory_store 
                           if m['user_id'] == user_id and m['timestamp'] >= cutoff_time]
                filtered.sort(key=lambda x: x['timestamp'], reverse=True)
                
                context = []
                for m in filtered[:20]:
                    context.append(f"User: {m['query']}")
                    context.append(f"Assistant: {m['response'][:200]}...")
                    
                return context
                
        except Exception as e:
            logger.error(f"Error retrieving user context: {e}")
            return []

    def create_new_thread_id(self, user_id: str) -> str:
        """Create a new thread ID"""
        return str(uuid.uuid4())

    def close(self):
        """Close database connection"""
        if hasattr(self, 'client'):
            self.client.close()

# ============================================================================
# SEARCH SERVICE
# ============================================================================

class SearchService:
    """Tavily-based search service"""
    
    def __init__(self):
        try:
            self.client = TavilyClient(api_key=Settings.TAVILY_API_KEY)
            logger.info(" Tavily client initialized")
        except Exception as e:
            logger.error(f"❌ Tavily initialization failed: {e}")
            self.client = None

    def search_related_topics(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for related topics"""
        if not self.client:
            logger.warning("Tavily client not available, returning mock results")
            return self._get_mock_search_results(query)
            
        try:
            logger.info(f"Searching for: {query}")
            
            # Main search
            main_results = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=5
            )
            
            all_results = list(main_results.get('results', []))
            
            # Related searches
            related_queries = self._generate_related_queries(query)
            
            for related_query in related_queries[:3]:
                try:
                    related_results = self.client.search(
                        query=related_query,
                        search_depth="basic",
                        max_results=2
                    )
                    all_results.extend(related_results.get('results', []))
                except Exception as e:
                    logger.warning(f"Error searching '{related_query}': {e}")
                    continue
            
            # Remove duplicates
            seen_urls = set()
            unique_results = []
            
            for result in all_results:
                url = result.get('url', '')
                if url not in seen_urls:
                    seen_urls.add(url)
                    unique_results.append(result)
                    
                if len(unique_results) >= max_results:
                    break
            
            logger.info(f"Found {len(unique_results)} unique search results")
            return unique_results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return self._get_mock_search_results(query)

    def _generate_related_queries(self, query: str) -> List[str]:
        """Generate related search queries"""
        return [
            f"how to {query}",
            f"best practices {query}",
            f"{query} guide",
            f"latest trends {query}",
            f"{query} examples",
            f"{query} tips"
        ]

    def _get_mock_search_results(self, query: str) -> List[Dict[str, Any]]:
        """Mock search results for testing"""
        return [
            {
                'title': f'Complete Guide to {query.title()}',
                'content': f'This is a comprehensive guide about {query}. It covers all the essential aspects...',
                'url': 'https://example.com/guide',
                'score': 0.9
            },
            {
                'title': f'Best Practices for {query.title()}',
                'content': f'Learn the best practices and tips for {query}. Industry experts share their insights...',
                'url': 'https://example.com/best-practices',
                'score': 0.8
            }
        ]

# ============================================================================
# LLM SERVICE
# ============================================================================

class LLMService:
    """Gemini-based LLM service"""
    
    def __init__(self):
        try:
            genai.configure(api_key=Settings.GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemini-1.5-flash')

            logger.info(" Gemini model initialized")
        except Exception as e:
            logger.error(f"❌ Gemini initialization failed: {e}")
            self.model = None

    def classify_intent(self, query: str, context: List[str] = None) -> bool:
        """Classify if query is a blog request"""
        if not self.model:
            # Fallback classification
            blog_keywords = ['blog', 'article', 'write', 'create', 'generate', 'content', 'post', 'seo']
            return any(keyword in query.lower() for keyword in blog_keywords)
            
        try:
            context_str = "\n".join(context[-5:]) if context else ""
            
            prompt = f"""
            Context: {context_str}
            
            User Query: "{query}"
            
            Is this a request to generate a blog post or article?
            Consider words like: blog, article, write, create, generate, post, content, SEO
            
            Respond with only "YES" or "NO".
            
            Examples:
            "Hi" -> NO
            "Write a blog about AI" -> YES
            "Create an article on cooking" -> YES
            """
            
            response = self.model.generate_content(prompt)
            response_text = ""
            if hasattr(response, "text"):
                response_text = response.text.strip()
            elif isinstance(response, dict):
                response_text = response.get("output_text", "").strip()
            else:
                response_text = str(response).strip()
            
            # Parse the response to determine if it's a blog request
            return "YES" in response_text.upper()
            
        except Exception as e:
            logger.error(f"Error in intent classification: {e}")
            # Fallback
            blog_keywords = ['blog', 'article', 'write', 'create', 'generate', 'content', 'post']
            return any(keyword in query.lower() for keyword in blog_keywords)

    def generate_casual_response(self, query: str, context: List[str] = None) -> str:
        """Generate casual response"""
        if not self.model:
            return self._get_fallback_casual_response(query)
            
        try:
            context_str = "\n".join(context[-3:]) if context else ""
            
            prompt = f"""
            Context: {context_str}
            
            User: {query}
            
            Generate a friendly, helpful response. Be conversational and natural.
            Keep it concise and engaging.
            """
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error generating casual response: {e}")
            return self._get_fallback_casual_response(query)

    def generate_blog_content(self, query: str, search_results: List[Dict[str, Any]], 
                            context: List[str] = None) -> BlogContent:
        """Generate blog content"""
        if not self.model:
            return self._get_fallback_blog_content(query)
            
        try:
            search_context = self._prepare_search_context(search_results)
            context_str = "\n".join(context[-5:]) if context else ""
            
            prompt = f"""
            Context: {context_str}
            
            User Request: {query}
            
            Research Data: {search_context}
            
            Create a comprehensive, SEO-friendly blog post. Make it human, engaging, and actionable.
            Include proper structure with headings. Aim for 1200+ words.
            
            Return JSON format:
            {{
                "title": "SEO-friendly title",
                "content": "Full blog content with markdown",
                "meta_description": "160-character description",
                "tags": ["tag1", "tag2", "tag3"],
                "seo_keywords": ["keyword1", "keyword2", "keyword3"],
                "word_count": estimated_word_count
            }}
            """
            
            response = self.model.generate_content(prompt)
            
            try:
                blog_data = json.loads(response.text.strip())
                return BlogContent(**blog_data)
            except json.JSONDecodeError:
                return self._parse_fallback_blog(query, response.text)
                
        except Exception as e:
            logger.error(f"Error generating blog: {e}")
            return self._get_fallback_blog_content(query)

    def _prepare_search_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Prepare search context"""
        context_parts = []
        
        for i, result in enumerate(search_results[:8], 1):
            title = result.get('title', 'No title')
            content = result.get('content', result.get('snippet', 'No content'))
            
            context_parts.append(f"""
            Source {i}:
            Title: {title}
            Content: {content[:500]}...
            """)
        
        return "\n".join(context_parts)

    def _get_fallback_casual_response(self, query: str) -> str:
        """Fallback casual responses"""
        responses = {
            'hi': 'Hello! How can I help you today?',
            'hello': 'Hi there! What can I do for you?',
            'how are you': "I'm doing great, thank you! How can I assist you?",
            'thanks': "You're welcome! Is there anything else I can help you with?"
        }
        
        query_lower = query.lower().strip()
        for key, response in responses.items():
            if key in query_lower:
                return response
                
        return "Hello! I'm here to help you with blog generation and casual conversation. What would you like to do?"

    def _get_fallback_blog_content(self, query: str) -> BlogContent:
        """Fallback blog content"""
        return BlogContent(
            title=f"Complete Guide to {query.title()}",
            content=f"""# {query.title()}: A Comprehensive Guide

## Introduction

Welcome to this comprehensive guide about {query}. In this article, we'll explore everything you need to know to get started and excel in this topic.

## What is {query.title()}?

{query.title()} is an important topic that affects many aspects of our daily lives. Understanding the fundamentals is crucial for anyone looking to improve their knowledge and skills in this area.

## Key Benefits

Here are some key benefits of understanding {query}:

- **Improved Knowledge**: Gain deeper insights into the subject matter
- **Practical Applications**: Learn how to apply concepts in real-world scenarios  
- **Personal Growth**: Develop new skills and competencies
- **Professional Development**: Enhance your career prospects

## Getting Started

To begin your journey with {query}, consider these essential steps:

### Step 1: Understanding the Basics
Start by familiarizing yourself with the fundamental concepts and terminology.

### Step 2: Practical Application
Apply what you've learned through hands-on practice and real-world examples.

### Step 3: Continuous Learning
Stay updated with the latest trends and developments in this field.

## Best Practices

Follow these best practices to maximize your success:

1. **Stay Consistent**: Regular practice leads to mastery
2. **Seek Feedback**: Learn from others' experiences and insights
3. **Document Progress**: Keep track of your learning journey
4. **Network**: Connect with others interested in the same topic

## Common Challenges and Solutions

Every journey has its challenges. Here are some common ones and how to overcome them:

- **Challenge**: Information overload
  - **Solution**: Start with basics and gradually build complexity

- **Challenge**: Lack of practical experience
  - **Solution**: Seek hands-on opportunities and practice regularly

## Conclusion

Understanding {query} is a valuable investment in your personal and professional development. By following the guidance in this article, you'll be well-equipped to succeed in this area.

Remember, learning is a continuous process. Stay curious, keep practicing, and don't hesitate to seek help when needed.

## Additional Resources

For further learning, consider exploring:
- Online courses and tutorials
- Books and research papers
- Community forums and discussions
- Professional workshops and seminars

Start your journey today and unlock the potential that {query} has to offer!
            """,
            meta_description=f"Complete guide to {query}. Learn everything you need to know with practical tips and expert insights.",
            tags=[query.lower(), 'guide', 'tutorial', 'tips'],
            seo_keywords=[query.lower(), 'guide', 'how to'],
            word_count=500
        )

    def _parse_fallback_blog(self, query: str, raw_content: str) -> BlogContent:
        """Parse blog when JSON fails"""
        return BlogContent(
            title=f"Guide to {query.title()}",
            content=raw_content,
            meta_description=f"Comprehensive guide about {query}",
            tags=[query.lower(), 'guide'],
            seo_keywords=[query.lower()],
            word_count=len(raw_content.split())
        )

# ============================================================================
# BLOG GENERATION AGENT
# ============================================================================

class BlogGenerationAgent:
    """Main blog generation agent using LangGraph"""
    
    def __init__(self):
        self.search_service = SearchService()
        self.llm_service = LLMService()
        self.memory_manager = MemoryManager()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("load_memory", self._load_memory)
        workflow.add_node("classify_intent", self._classify_intent)
        workflow.add_node("generate_casual_response", self._generate_casual_response)
        workflow.add_node("search_topics", self._search_topics)
        workflow.add_node("generate_blog", self._generate_blog)
        workflow.add_node("save_memory", self._save_memory)
        workflow.add_node("handle_error", self._handle_error)

        # Define workflow
        workflow.set_entry_point("load_memory")
        workflow.add_edge("load_memory", "classify_intent")
        
        workflow.add_conditional_edges(
            "classify_intent",
            self._route_intent,
            {
                "casual": "generate_casual_response",
                "blog": "search_topics",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("generate_casual_response", "save_memory")
        workflow.add_edge("search_topics", "generate_blog")
        workflow.add_edge("generate_blog", "save_memory")
        workflow.add_edge("save_memory", END)
        workflow.add_edge("handle_error", END)

        return workflow.compile()

    async def process_query(self, query: str, user_id: str = None, thread_id: str = None) -> Dict[str, Any]:
        """Process user query"""
        try:
            if not user_id:
                user_id = get_current_user()
                
            if not thread_id:
                thread_id = self.memory_manager.create_new_thread_id(user_id)

            # Initialize state
            initial_state = AgentState(
                query=query,
                user_id=user_id,
                thread_id=thread_id
            )

            # Run workflow
            final_state = await self.graph.ainvoke(initial_state)

            # The compiled workflow may return either an AgentState instance
            # or a plain dict depending on the langgraph/runtime version.
            # Normalize both cases to a consistent dict response.
            if isinstance(final_state, dict):
                response = final_state.get('response', '')
                is_blog = final_state.get('is_blog_request', False)
                blog_content_raw = final_state.get('blog_content')
                error = final_state.get('error')

                if isinstance(blog_content_raw, dict):
                    blog_content = blog_content_raw
                elif blog_content_raw is None:
                    blog_content = None
                else:
                    # support dataclass/object
                    blog_content = getattr(blog_content_raw, '__dict__', None)
            else:
                response = getattr(final_state, 'response', '')
                is_blog = getattr(final_state, 'is_blog_request', False)
                blog_obj = getattr(final_state, 'blog_content', None)
                blog_content = blog_obj.__dict__ if blog_obj else None
                error = getattr(final_state, 'error', None)

            return {
                "response": response,
                "thread_id": thread_id,
                "is_blog_request": is_blog,
                "blog_content": blog_content,
                "error": error
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "error": str(e),
                "response": "I apologize, but I encountered an error. Please try again.",
                "thread_id": thread_id or "unknown"
            }

    def _load_memory(self, state: AgentState) -> AgentState:
        """Load conversation memory"""
        try:
            thread_history = self.memory_manager.get_thread_history(
                state.user_id, state.thread_id, limit=5
            )
            
            user_context = self.memory_manager.get_user_context(
                state.user_id, hours=24
            )
            
            memory_context = []
            for memory in thread_history:
                memory_context.append(f"User: {memory.query}")
                memory_context.append(f"Assistant: {memory.response}")
            
            memory_context.extend(user_context[-10:])
            state.memory_context = memory_context
            
        except Exception as e:
            logger.error(f"Error loading memory: {e}")
            state.memory_context = []
        
        return state

    def _classify_intent(self, state: AgentState) -> AgentState:
        """Classify user intent"""
        try:
            state.is_blog_request = self.llm_service.classify_intent(
                state.query, state.memory_context
            )
        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            state.error = str(e)
        
        return state

    def _generate_casual_response(self, state: AgentState) -> AgentState:
        """Generate casual response"""
        try:
            state.response = self.llm_service.generate_casual_response(
                state.query, state.memory_context
            )
        except Exception as e:
            logger.error(f"Error generating casual response: {e}")
            state.response = "Hello! How can I help you today?"
        
        return state

    def _search_topics(self, state: AgentState) -> AgentState:
        """Search for topics"""
        try:
            state.search_results = self.search_service.search_related_topics(
                state.query, max_results=10
            )
            logger.info(f"Search completed: {len(state.search_results)} results")
        except Exception as e:
            logger.error(f"Error searching: {e}")
            state.search_results = []
        
        return state

    def _generate_blog(self, state: AgentState) -> AgentState:
        """Generate blog content"""
        try:
            state.blog_content = self.llm_service.generate_blog_content(
                state.query, state.search_results, state.memory_context
            )
            
            # Format response
            blog = state.blog_content
            state.response = f"""# {blog.title}

{blog.content}

---
**📊 Blog Statistics:**
- **Word Count:** {blog.word_count}
- **Meta Description:** {blog.meta_description}
- **Tags:** {', '.join(blog.tags)}
- **SEO Keywords:** {', '.join(blog.seo_keywords)}
- **Search Sources:** {len(state.search_results)} articles researched
"""
            
        except Exception as e:
            logger.error(f"Error generating blog: {e}")
            state.response = "I apologize, but I couldn't generate the blog. Please try again."
            state.error = str(e)
        
        return state

    def _save_memory(self, state: AgentState) -> AgentState:
        """Save conversation to memory"""
        try:
            memory = ConversationMemory(
                user_id=state.user_id,
                thread_id=state.thread_id,
                query=state.query,
                response=state.response,
                metadata={
                    "is_blog_request": state.is_blog_request,
                    "has_blog_content": state.blog_content is not None,
                    "search_results_count": len(state.search_results),
                    "word_count": state.blog_content.word_count if state.blog_content else 0
                }
            )
            
            self.memory_manager.save_conversation(memory)
            
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
        
        return state

    def _handle_error(self, state: AgentState) -> AgentState:
        """Handle errors"""
        if not state.response:
            state.response = "I apologize for the error. Please try again."
        return state

    def _route_intent(self, state: AgentState) -> str:
        """Route based on intent"""
        if state.error:
            return "error"
        elif state.is_blog_request:
            return "blog"
        else:
            return "casual"

    def close(self):
        """Cleanup resources"""
        self.memory_manager.close()

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

class BlogAgentCLI:
    """Command line interface for the blog agent"""
    
    def __init__(self):
        self.agent = None
        self.current_thread = None

    def display_banner(self):
        """Display welcome banner"""
        print("\n" + "="*60)
        print("🤖 BLOG GENERATION AI AGENT")
        print("="*60)
        print("📝 Generate SEO-friendly blogs with AI research")
        print("💬 Chat casually or request comprehensive articles")
        print("🔍 Powered by Gemini AI + Tavily Search")
        print("="*60)

    def display_help(self):
        """Display help information"""
        print("\n📋 COMMANDS:")
        print("  /help    - Show this help")
        print("  /new     - Start new conversation thread")
        print("  /exit    - Exit the application")
        print("  /status  - Show system status")
        print("\n💡 EXAMPLES:")
        print("  • Hi there!")
        print("  • Write a blog about sustainable living")
        print("  • Create an article on Python programming")
        print("  • Generate content about digital marketing trends")

    def display_status(self):
        """Display system status"""
        print("\n SYSTEM STATUS:")
        print(f"   Agent: {'Ready' if self.agent else 'Not initialized'}")
        print(f"   Thread: {self.current_thread[:8] + '...' if self.current_thread else 'None'}")
        print(f"   Gemini API: {' Connected' if Settings.GEMINI_API_KEY else '❌ Missing'}")
        print(f"   Tavily API: {' Connected' if Settings.TAVILY_API_KEY else '❌ Missing'}")
        print(f"   MongoDB: {' Connected' if Settings.MONGODB_URL else '❌ Not configured'}")

    async def initialize_agent(self):
        """Initialize the agent"""
        print("\n Initializing Blog Generation Agent...")
        
        if not Settings.validate():
            print("\n❌ Environment setup incomplete!")
            print("\n📋 Setup Instructions:")
            print("1. Create a .env file with:")
            print("   GEMINI_API_KEY=your_key_here")
            print("   TAVILY_API_KEY=your_key_here") 
            print("   MONGODB_URL=mongodb://localhost:27017/blog_agent")
            print("\n2. Install requirements:")
            print("   pip install langgraph google-generativeai tavily-python pymongo python-dotenv")
            return False
            
        try:
            self.agent = BlogGenerationAgent()
            self.current_thread = None
            print(" Agent initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ Agent initialization failed: {e}")
            return False

    async def process_command(self, user_input: str) -> bool:
        """Process user commands"""
        command = user_input.strip().lower()
        
        if command == '/exit':
            print("\n👋 Goodbye! Thanks for using Blog Generation Agent!")
            return False
        elif command == '/help':
            self.display_help()
            return True
        elif command == '/new':
            self.current_thread = None
            print("🆕 Started new conversation thread")
            return True
        elif command == '/status':
            self.display_status()
            return True
        else:
            # Process as regular query
            await self.process_query(user_input)
            return True

    async def process_query(self, query: str):
        """Process user query through the agent"""
        try:
            print("\n🤔 Processing your request...")
            
            result = await self.agent.process_query(
                query=query,
                thread_id=self.current_thread
            )
            
            # Update current thread
            self.current_thread = result.get('thread_id')
            
            # Display result
            if result.get('error'):
                print(f"\n❌ Error: {result['error']}")
            else:
                response = result.get('response', '')
                is_blog = result.get('is_blog_request', False)
                
                if is_blog:
                    print("\n📝 BLOG GENERATED:")
                    print("=" * 50)
                else:
                    print("\n💬 RESPONSE:")
                    print("-" * 30)
                
                print(response)
                
                if is_blog:
                    print("\n" + "=" * 50)
                    blog_content = result.get('blog_content')
                    if blog_content:
                        print(f"📊 Generated {blog_content['word_count']} words")
                        print(f"🏷️  Tags: {', '.join(blog_content['tags'])}")
                
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")

    async def run(self):
        """Main CLI loop"""
        self.display_banner()
        
        if not await self.initialize_agent():
            return
            
        self.display_help()
        
        print("\n🎯 Ready! Type your message or command...")
        
        try:
            while True:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                    
                should_continue = await self.process_command(user_input)
                if not should_continue:
                    break
                    
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! (Ctrl+C pressed)")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
        finally:
            if self.agent:
                self.agent.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main function"""
    cli = BlogAgentCLI()
    await cli.run()

def setup_environment():
    """Setup environment file if it doesn't exist"""
    env_file = ".env"
    if not os.path.exists(env_file):
        print("🔧 Creating .env file template...")
        with open(env_file, 'w') as f:
            f.write("""# Blog Generation Agent - Environment Variables
# Get your API keys from:
# Gemini: https://makersuite.google.com/app/apikey
# Tavily: https://tavily.com

GEMINI_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
MONGODB_URL=mongodb://localhost:27017/blog_agent

# Optional: Set to 'true' to enable debug logging
DEBUG=false
""")
        print(" Created .env file template")
        print("📝 Please edit .env file with your API keys")
        return False
    return True

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'google.generativeai',
        'tavily',
        'pymongo',
        'langgraph',
        'dotenv'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing required packages:")
        for package in missing:
            print(f"   - {package}")
        print("\n📦 Install with:")
        print("   pip install langgraph google-generativeai tavily-python pymongo python-dotenv")
        return False
    
    return True

if __name__ == "__main__":
    print(" Blog Generation AI Agent - Single File Version")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        exit(1)
    
    # Setup environment
    if not setup_environment():
        exit(1)
        
    # Run the agent
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"❌ Failed to start agent: {e}")
        exit(1)