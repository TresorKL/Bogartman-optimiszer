import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import io

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Bogart Man Inventory Optimizer",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E4057;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1E3A8A;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .input-section {
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #E2E8F0;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .analyze-button {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: bold;
        cursor: pointer;
        width: 100%;
        margin-top: 1rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: bold;
        width: 100%;
        height: 60px;
    }
</style>
""", unsafe_allow_html=True)

# Product categories and their historical sales percentages
PRODUCT_CATEGORIES = {
    'T-shirts': {'percentage': 16, 'avg_price': 1900},
    'Jeans': {'percentage': 8, 'avg_price': 2900},
    'Cargo Pants': {'percentage': 7, 'avg_price': 2800},
    'Jackets': {'percentage': 4, 'avg_price': 5000},
    'Hoodies': {'percentage': 6, 'avg_price': 2900},
    'Sweaters': {'percentage': 4, 'avg_price': 3200},
    'Golf Shirts': {'percentage': 13, 'avg_price': 2500},
    'Tracksuits': {'percentage': 9, 'avg_price': 6900},
    'Shirts': {'percentage': 5, 'avg_price': 2200},
    'Blazers': {'percentage': 3, 'avg_price': 4500},
    'Suits': {'percentage': 6, 'avg_price': 6500},
    'Knitted Jerseys': {'percentage': 2, 'avg_price': 3000},
    'Ties': {'percentage': 1, 'avg_price': 800},
    'Pendants': {'percentage': 0.5, 'avg_price': 1200},
    'Boots': {'percentage': 2, 'avg_price': 3500},
    'Sneakers': {'percentage': 5, 'avg_price': 2800},
    'Sandals': {'percentage': 1, 'avg_price': 1500},
    'Formal shoes': {'percentage': 2, 'avg_price': 3200},
    'Colognes': {'percentage': 1, 'avg_price': 1800},
    'Caps & Hats': {'percentage': 2, 'avg_price': 900},
    'Belts': {'percentage': 1, 'avg_price': 1200},
    'Teddy Bears': {'percentage': 1, 'avg_price': 800},
    'Cooler Boxes': {'percentage': 1, 'avg_price': 1500}
}

# Define all classes first
class AIStrategicAdvisor:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.context = """
        You are a senior retail strategy consultant specializing in luxury men's fashion and inventory optimization. 
        You're analyzing data for Bogart Man, a premium South African men's clothing brand that targets 
        sophisticated, successful men. The brand operates multiple stores and sources products from China.
        
        Key Business Context:
        - Bogart Man is a luxury brand with premium pricing
        - Target market: Successful, ambitious men who value quality and style
        - Multi-store operation across South Africa
        - Products sourced from China with significant markups
        - 23 product categories with varying performance levels
        - Historical sales data shows T-shirts (16%) and Golf Shirts (13%) as top performers
        
        Your role is to provide strategic, actionable recommendations based on the financial analysis.
        Be specific, practical, and consider both opportunities and risks.
        Return response as JSON array with objects containing: category, insight, recommendation, priority (HIGH/MEDIUM/LOW)
        """
    
    def generate_recommendations(self, analysis_data):
        """Generate AI-powered strategic recommendations using OpenAI"""
        
        if not self.api_key:
            self.api_key = os.environ.get('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY')
            
        if not self.api_key:
            return None
            
        return self.call_openai(analysis_data)
    
    def call_openai(self, analysis_data):
        """Call OpenAI GPT-4 API for recommendations"""
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            
            prompt = f"""
            {self.context}
            
            BUSINESS ANALYSIS DATA:
            {json.dumps(analysis_data, indent=2)}
            
            Based on this comprehensive analysis, provide strategic recommendations covering:
            
            1. INVESTMENT STRATEGY: Comment on the budget allocation and ROI potential
            2. PRODUCT MIX OPTIMIZATION: Which categories to focus on or avoid
            3. OPERATIONAL EFFICIENCY: Store distribution and inventory turnover insights
            4. RISK ASSESSMENT: Potential challenges and mitigation strategies
            5. GROWTH OPPORTUNITIES: Specific actions to maximize profitability
            6. MARKET POSITIONING: How this strategy aligns with Bogart Man's luxury brand
            """
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a senior retail strategy consultant. Return JSON array format with category, insight, recommendation, and priority fields."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            try:
                return json.loads(response.choices[0].message.content)
            except:
                return self.parse_text_to_recommendations(response.choices[0].message.content)
                
        except Exception as e:
            return None
    
    def parse_text_to_recommendations(self, text):
        """Parse text response into recommendation format"""
        recommendations = []
        sections = text.split('\n\n')
        
        for i, section in enumerate(sections[:6]):
            if section.strip():
                recommendations.append({
                    "category": f"📊 Strategic Insight {i+1}",
                    "insight": "AI-Generated Analysis",
                    "recommendation": section.strip(),
                    "priority": ["HIGH", "MEDIUM", "LOW"][i % 3]
                })
        
        return recommendations if recommendations else None

class InventoryOptimizer:
    def __init__(self):
        self.markup_multiplier = 7  # 600% markup = 7x multiplier
        
    def calculate_supplier_cost(self, retail_price):
        """Calculate optimal supplier cost based on 600% markup rule"""
        return retail_price / self.markup_multiplier
    
    def calculate_weighted_average_cost(self, categories):
        """Calculate weighted average supplier cost across all categories"""
        total_weight = sum(cat['percentage'] for cat in categories.values())
        weighted_cost = sum(
            (cat['percentage'] / total_weight) * self.calculate_supplier_cost(cat['avg_price'])
            for cat in categories.values()
        )
        return weighted_cost
    
    def calculate_inventory_requirements(self, budget, markup_percent, turnaround_days, num_stores, target_revenue=None):
        """Main optimization formula"""
        try:
            markup_decimal = markup_percent / 100
            
            # Calculate basic metrics
            avg_supplier_cost = self.calculate_weighted_average_cost(PRODUCT_CATEGORIES)
            
            # If target revenue is not provided, calculate based on budget and markup
            if target_revenue is None:
                target_revenue = budget * (1 + markup_decimal)
            
            # Calculate inventory turnover cycles needed
            annual_days = 365
            turnover_cycles = annual_days / turnaround_days
            
            # Calculate total products needed
            total_products = int(budget / avg_supplier_cost) if avg_supplier_cost > 0 else 0
            
            # Adjust for store distribution and turnover
            products_per_store = total_products / num_stores if num_stores > 0 else 0
            
            # Calculate category-specific requirements
            category_breakdown = {}
            for category, data in PRODUCT_CATEGORIES.items():
                category_budget = budget * (data['percentage'] / 100)
                category_supplier_cost = self.calculate_supplier_cost(data['avg_price'])
                category_quantity = int(category_budget / category_supplier_cost) if category_supplier_cost > 0 else 0
                
                category_breakdown[category] = {
                    'budget_allocation': category_budget,
                    'supplier_cost': category_supplier_cost,
                    'quantity': category_quantity,
                    'retail_price': data['avg_price'],
                    'total_retail_value': category_quantity * data['avg_price'],
                    'profit_per_unit': data['avg_price'] - category_supplier_cost
                }
            
            result = {
                'total_products': total_products,
                'products_per_store': products_per_store,
                'avg_supplier_cost': avg_supplier_cost,
                'category_breakdown': category_breakdown,
                'projected_revenue': sum(cat['total_retail_value'] for cat in category_breakdown.values()),
                'total_profit': sum(cat['quantity'] * cat['profit_per_unit'] for cat in category_breakdown.values()),
                'turnover_cycles': turnover_cycles
            }
            
            return result
            
        except Exception as e:
            return None

def display_results(results, budget, markup_percent, turnaround_days, num_stores, target_revenue):
    """Display all results and analysis"""
    
    # Basic safety check
    if not results:
        st.error("❌ No results to display")
        return
        
    # Extract data safely
    total_products = results.get('total_products', 0)
    avg_supplier_cost = results.get('avg_supplier_cost', 0)
    products_per_store = results.get('products_per_store', 0)
    projected_revenue = results.get('projected_revenue', 0)
    total_profit = results.get('total_profit', 0)
    turnover_cycles = results.get('turnover_cycles', 0)
    category_breakdown = results.get('category_breakdown', {})
    
    # Main dashboard
    st.markdown('<h2 class="section-header">📊 Financial Dashboard</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🛍️ Total Products</h3>
            <h2>{total_products:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💵 Avg Supplier Cost</h3>
            <h2>R{avg_supplier_cost:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Products/Store</h3>
            <h2>{products_per_store:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💰 Projected Revenue</h3>
            <h2>R{projected_revenue:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Financial Summary
    st.markdown('<h2 class="section-header">💼 Financial Summary</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("💸 Total Investment", f"R{budget:,.0f}")
        st.metric("📈 Total Profit", f"R{total_profit:,.0f}")
        st.metric("🔄 Turnover Cycles/Year", f"{turnover_cycles:.1f}")
    
    with col2:
        profit_margin = (total_profit / budget) * 100 if budget > 0 else 0
        st.metric("📊 ROI %", f"{profit_margin:.1f}%")
        st.metric("⚡ Revenue/Investment Ratio", f"{projected_revenue/budget:.1f}x" if budget > 0 else "0x")
        
        if target_revenue:
            markup_decimal = markup_percent / 100
            monthly_sales_capacity = budget * (1 + markup_decimal) / num_stores if num_stores > 0 else 0
            total_monthly_capacity = monthly_sales_capacity * num_stores
            months_needed = target_revenue / total_monthly_capacity if total_monthly_capacity > 0 else 0
            days_needed = months_needed * 30
            st.metric("⏰ Days to Target", f"{days_needed:.0f} days")
    
    # Product Category Breakdown
    st.markdown('<h2 class="section-header">📋 Product Category Breakdown</h2>', unsafe_allow_html=True)
    
    # Add explanation about the data source
    st.info("""
    📊 **About This Analysis**: The category breakdown below is based on Bogart Man's actual historical sales data 
    over the past 5 years. Each category's budget allocation is calculated using your proven sales percentages 
    (e.g., T-shirts 16%, Golf Shirts 13%, Tracksuits 9%) combined with current retail prices and the 600% markup strategy. 
    This ensures recommendations are grounded in your real business performance.
    """)
    
    # Create DataFrame for category data
    if category_breakdown:
        category_df = pd.DataFrame([
            {
                'Category': category,
                'Sales %': PRODUCT_CATEGORIES.get(category, {}).get('percentage', 0),
                'Budget Allocation': f"R{data.get('budget_allocation', 0):,.0f}",
                'Quantity': data.get('quantity', 0),
                'Supplier Cost': f"R{data.get('supplier_cost', 0):,.0f}",
                'Retail Price': f"R{data.get('retail_price', 0):,.0f}",
                'Total Retail Value': f"R{data.get('total_retail_value', 0):,.0f}",
                'Profit per Unit': f"R{data.get('profit_per_unit', 0):,.0f}"
            }
            for category, data in category_breakdown.items()
        ])
        
        st.dataframe(category_df, use_container_width=True)
    else:
        st.warning("⚠️ Category breakdown data not available")
    
    # Visualizations
    st.markdown('<h2 class="section-header">📊 Visual Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Budget allocation pie chart
        if category_breakdown:
            fig_pie = px.pie(
                values=[data.get('budget_allocation', 0) for data in category_breakdown.values()],
                names=list(category_breakdown.keys()),
                title="Budget Allocation by Category"
            )
            fig_pie.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.05
                ),
                margin=dict(l=20, r=150, t=50, b=20)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("📊 Chart data not available")
    
    with col2:
        # Quantity distribution
        if category_breakdown:
            fig_bar = px.bar(
                x=list(category_breakdown.keys()),
                y=[data.get('quantity', 0) for data in category_breakdown.values()],
                title="Product Quantities by Category"
            )
            fig_bar.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("📊 Chart data not available")
    
    # Profitability analysis
    st.markdown('<h2 class="section-header">💹 Profitability Analysis</h2>', unsafe_allow_html=True)
    
    # Create profitability DataFrame
    if category_breakdown:
        profit_df = pd.DataFrame([
            {
                'Category': category,
                'Investment': data.get('budget_allocation', 0),
                'Revenue': data.get('total_retail_value', 0),
                'Profit': data.get('quantity', 0) * data.get('profit_per_unit', 0),
                'ROI %': ((data.get('quantity', 0) * data.get('profit_per_unit', 0)) / data.get('budget_allocation', 1)) * 100
            }
            for category, data in category_breakdown.items()
            if data.get('budget_allocation', 0) > 0
        ])
        
        if not profit_df.empty:
            fig_profit = px.scatter(
                profit_df,
                x='Investment',
                y='Profit',
                size='ROI %',
                hover_data=['Category'],
                title="Investment vs Profit by Category (Size = ROI %)"
            )
            st.plotly_chart(fig_profit, use_container_width=True)
        else:
            st.info("📊 Profitability chart data not available")
    else:
        st.info("📊 Profitability analysis not available")
    
    # AI-Powered Strategic Recommendations
    st.markdown('<h2 class="section-header">🤖 AI Strategic Recommendations</h2>', unsafe_allow_html=True)
    
    # Check if OpenAI API key is available
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        st.warning("🤖 **AI Analysis Not Available**")
        st.info("""
        **To enable AI-powered strategic recommendations:**
        
        1. **Set up your OpenAI API key** in your environment variables
        2. **Create a .env file** with: `OPENAI_API_KEY=your_key_here`
        3. **Restart the application** to load the new configuration
        
        **What AI Analysis Provides:**
        - 💰 Investment Strategy Analysis
        - 🎯 Product Mix Optimization  
        - ⚡ Operational Efficiency Insights
        - ⚠️ Risk Assessment
        - 🚀 Growth Opportunities
        - 👔 Market Positioning Advice
        """)
    else:
        try:
            # Only create AI advisor if we have the key
            ai_advisor = AIStrategicAdvisor()
            analysis_data = {
                'budget': budget,
                'markup_percent': markup_percent,
                'turnaround_days': turnaround_days,
                'num_stores': num_stores,
                'total_products': total_products,
                'avg_supplier_cost': avg_supplier_cost,
                'projected_revenue': projected_revenue,
                'total_profit': total_profit,
                'turnover_cycles': turnover_cycles,
                'category_breakdown': category_breakdown,
                'roi_percentage': (total_profit / budget) * 100 if budget > 0 else 0
            }
            
            # Generate AI recommendations
            with st.spinner('🧠 AI Advisor analyzing your strategy...'):
                ai_recommendations = ai_advisor.generate_recommendations(analysis_data)
            
            # Display AI recommendations
            if ai_recommendations:
                st.success("✅ AI Analysis Complete using GPT-4")
                
                for i, rec in enumerate(ai_recommendations):
                    priority_color = {
                        'HIGH': '#DC2626',
                        'MEDIUM': '#D97706', 
                        'LOW': '#059669'
                    }.get(rec.get('priority', 'MEDIUM'), '#6B7280')
                    
                    with st.container():
                        st.markdown(f"""
                        <div style="
                            border-left: 4px solid {priority_color};
                            background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
                            padding: 1.5rem;
                            margin: 1rem 0;
                            border-radius: 8px;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                        ">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <h3 style="color: #1E293B; margin: 0; font-size: 1.2rem;">{rec.get('category', 'Strategic Insight')}</h3>
                                <span style="
                                    background: {priority_color};
                                    color: white;
                                    padding: 0.25rem 0.75rem;
                                    border-radius: 12px;
                                    font-size: 0.75rem;
                                    font-weight: bold;
                                ">{rec.get('priority', 'MEDIUM')} PRIORITY</span>
                            </div>
                            <p style="color: #475569; font-weight: 600; margin: 0.5rem 0; font-size: 0.95rem;">
                                💡 <strong>{rec.get('insight', 'Analysis Insight')}</strong>
                            </p>
                            <p style="color: #334155; margin: 0; line-height: 1.6; font-size: 0.95rem;">
                                {rec.get('recommendation', 'Strategic recommendation')}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Add AI Analysis Summary
                st.markdown("### 📊 AI Analysis Summary")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    high_priority = len([r for r in ai_recommendations if r.get('priority') == 'HIGH'])
                    st.metric("🔥 High Priority Actions", high_priority)
                
                with col2:
                    medium_priority = len([r for r in ai_recommendations if r.get('priority') == 'MEDIUM'])
                    st.metric("⚡ Medium Priority Actions", medium_priority)
                
                with col3:
                    confidence_score = min(95, max(60, (total_profit / budget * 100) / 5)) if budget > 0 else 60
                    st.metric("🎯 Strategy Confidence", f"{confidence_score:.0f}%")
                    
            else:
                st.warning("🤖 AI Analysis Unavailable")
                st.info("Please check your OpenAI API key configuration")
                
        except Exception as e:
            st.warning("🤖 AI Analysis Error")
            st.info(f"AI functionality temporarily unavailable: {str(e)}")
    
    # Export functionality
    st.markdown('<h2 class="section-header">📤 Export Results</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Generate Detailed Report", use_container_width=True):
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'input_parameters': {
                    'budget': budget,
                    'markup_percent': markup_percent,
                    'turnaround_days': turnaround_days,
                    'num_stores': num_stores,
                    'target_revenue': target_revenue
                },
                'financial_summary': {
                    'total_products': total_products,
                    'avg_supplier_cost': avg_supplier_cost,
                    'projected_revenue': projected_revenue,
                    'total_profit': total_profit,
                    'roi_percentage': (total_profit / budget) * 100 if budget > 0 else 0,
                    'turnover_cycles': turnover_cycles
                },
                'category_breakdown': category_breakdown
            }
            
            st.json(report_data)
            st.success("📋 Comprehensive report generated!")
    
    with col2:
        if st.button("🔄 Analyze Again", use_container_width=True):
            st.rerun()

def main():
    st.markdown('<h1 class="main-header">🏢 Bogart Man Inventory Optimizer</h1>', unsafe_allow_html=True)
    
    # Input Section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("## 📊 Strategy Inputs")
    st.markdown("*Adjust your business parameters below - results update automatically*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        budget = st.number_input(
            "💰 Budget to Spend on Goods (ZAR)",
            min_value=10000,
            max_value=10000000,
            value=500000,
            step=10000,
            help="Total budget available for purchasing inventory from suppliers"
        )
        
        turnaround_days = st.number_input(
            "⏱️ Preferred Turnaround Time (Days)",
            min_value=30,
            max_value=365,
            value=90,
            step=15,
            help="How many days to sell through inventory"
        )
    
    with col2:
        markup_percent = st.slider(
            "📈 Markup/Profit Margin (%)",
            min_value=50,
            max_value=1000,
            value=600,
            step=10,
            help="Percentage markup on supplier cost (600% means 7x multiplier)"
        )
        
        num_stores = st.number_input(
            "🏪 Number of Stores",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            help="Number of stores to distribute inventory across"
        )
    
    # Optional target revenue
    use_target_revenue = st.checkbox("🎯 Set Specific Revenue Target")
    target_revenue = None
    if use_target_revenue:
        target_revenue = st.number_input(
            "Target Revenue (ZAR)",
            min_value=budget,
            max_value=budget * 20,
            value=int(budget * 2),
            step=10000
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Update Analysis Button
    if st.button("🚀 Update Analysis", use_container_width=True, type="primary"):
        with st.spinner("🔄 Updating your inventory strategy..."):
            st.rerun()
        st.balloons()
    
    st.markdown("---")
    
    # Initialize optimizer
    optimizer = InventoryOptimizer()
    
    # Always show results with current parameters
    try:
        results = optimizer.calculate_inventory_requirements(
            budget, markup_percent, turnaround_days, num_stores, target_revenue
        )
        
        if results:
            display_results(results, budget, markup_percent, turnaround_days, num_stores, target_revenue)
        else:
            st.error("❌ No results generated")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        # Create minimal working results as fallback
        fallback_results = {
            'total_products': int(budget / 400),
            'products_per_store': int(budget / 400 / num_stores),
            'avg_supplier_cost': 400,
            'category_breakdown': {cat: {
                'budget_allocation': budget * (data['percentage'] / 100),
                'supplier_cost': data['avg_price'] / 7,
                'quantity': int((budget * data['percentage'] / 100) / (data['avg_price'] / 7)),
                'retail_price': data['avg_price'],
                'total_retail_value': int((budget * data['percentage'] / 100) / (data['avg_price'] / 7)) * data['avg_price'],
                'profit_per_unit': data['avg_price'] - (data['avg_price'] / 7)
            } for cat, data in PRODUCT_CATEGORIES.items()},
            'projected_revenue': budget * 7,
            'total_profit': budget * 6,
            'turnover_cycles': 365 / turnaround_days
        }
        display_results(fallback_results, budget, markup_percent, turnaround_days, num_stores, target_revenue)

if __name__ == "__main__":
    main()