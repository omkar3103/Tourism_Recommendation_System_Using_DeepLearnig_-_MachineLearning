"""
Tourism Recommendation System - Enhanced Streamlit Frontend
Beautiful UI with animations and interactive features
Handles TensorFlow installation and Windows compatibility
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
import sys
import subprocess
import os

warnings.filterwarnings('ignore')

# ===== TENSORFLOW INSTALLATION CHECK =====
def check_and_install_tensorflow():
    """Check if TensorFlow is installed, install if not"""
    try:
        import tensorflow as tf
        return tf
    except ImportError:
        st.warning("TensorFlow not found. Installing... This may take a few minutes.")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow", "--quiet"])
            import tensorflow as tf
            st.success("✅ TensorFlow installed successfully!")
            return tf
        except Exception as e:
            st.error(f"Failed to install TensorFlow: {str(e)}")
            st.info("Please manually install: pip install tensorflow")
            return None

# Load TensorFlow
tf = check_and_install_tensorflow()
if tf is None:
    st.stop()

from tensorflow import keras
import time

# Check for Plotly
try:
    import plotly.graph_objects as go
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly", "--quiet"])
    import plotly.graph_objects as go

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Page Configuration
st.set_page_config(
    page_title="AI Tourism Recommender",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 10px;
        animation: fadeInDown 1s ease-in;
    }
    .subtitle {
        text-align: center;
        color: #f0f0f0;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    .recommendation-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin: 10px 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .recommendation-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.2);
    }
    .metric-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
</style>
""", unsafe_allow_html=True)


class TourismRecommenderApp:
    """Streamlit App for Tourism Recommendations"""
    
    def __init__(self, model_path, csv_path):
        self.model_path = model_path
        self.csv_path = csv_path
        self.model = None
        self.df_processed = None
        self.destination_features = None
        
        # Encoders
        self.user_encoder = LabelEncoder()
        self.dest_encoder = LabelEncoder()
        self.state_encoder = LabelEncoder()
        self.type_encoder = LabelEncoder()
        self.time_encoder = LabelEncoder()
        self.pref_encoder = LabelEncoder()
        self.feature_scaler = StandardScaler()
    
    def load_model(self):
        """Load pre-trained model with error handling"""
        try:
            if not os.path.exists(self.model_path):
                return None, f"Model file not found at: {self.model_path}"
            
            model = keras.models.load_model(
                self.model_path,
                custom_objects={
                    'mse': keras.losses.MeanSquaredError(),
                    'mae': keras.losses.MeanAbsoluteError(),
                },
                compile=False
            )
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae', 'mse', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
            )
            
            return model, None
        except Exception as e:
            return None, f"Error loading model: {str(e)}"
    
    def load_data(self):
        """Load and preprocess data with error handling"""
        try:
            if not os.path.exists(self.csv_path):
                return None, f"Data file not found at: {self.csv_path}"
            
            df = pd.read_csv(self.csv_path)
            
            # Preprocessing
            df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce').fillna(df['Rating'].median())
            df['ExperienceRating'] = pd.to_numeric(df['ExperienceRating'], errors='coerce').fillna(df['Rating'])
            df['NumberOfAdults'] = pd.to_numeric(df['NumberOfAdults'], errors='coerce').fillna(2)
            df['NumberOfChildren'] = pd.to_numeric(df['NumberOfChildren'], errors='coerce').fillna(0)
            df['Popularity'] = pd.to_numeric(df['Popularity'], errors='coerce').fillna(df['Popularity'].median())
            
            df['State'] = df['State'].fillna('Unknown')
            df['Type'] = df['Type'].fillna('General')
            df['BestTimeToVisit'] = df['BestTimeToVisit'].fillna('Year-round')
            df['Preferences'] = df['Preferences'].fillna('General Tourism')
            
            # Feature engineering
            df['total_travelers'] = df['NumberOfAdults'] + df['NumberOfChildren']
            df['has_children'] = (df['NumberOfChildren'] > 0).astype(int)
            df['children_ratio'] = df['NumberOfChildren'] / (df['total_travelers'] + 1e-8)
            df['is_family'] = (df['total_travelers'] >= 3).astype(int)
            
            df['pref_cleaned'] = df['Preferences'].str.lower().str.strip()
            df['is_adventure'] = df['pref_cleaned'].str.contains('adventure|trek|hiking', na=False).astype(int)
            df['is_cultural'] = df['pref_cleaned'].str.contains('cultural|heritage|historical', na=False).astype(int)
            
            return df, None
        except Exception as e:
            return None, f"Error loading data: {str(e)}"
    
    def fit_encoders(self, df):
        """Fit all encoders"""
        df['user_encoded'] = self.user_encoder.fit_transform(df['UserID'].astype(str))
        df['dest_encoded'] = self.dest_encoder.fit_transform(df['DestinationID_x'].astype(str))
        df['state_encoded'] = self.state_encoder.fit_transform(df['State'].astype(str))
        df['type_encoded'] = self.type_encoder.fit_transform(df['Type'].astype(str))
        df['time_encoded'] = self.time_encoder.fit_transform(df['BestTimeToVisit'].astype(str))
        df['pref_encoded'] = self.pref_encoder.fit_transform(df['pref_cleaned'].astype(str))
        
        # Aggregated features
        dest_stats = df.groupby('DestinationID_x').agg({
            'Rating': ['mean', 'count'],
            'NumberOfChildren': 'mean',
            'Popularity': 'mean'
        }).reset_index()
        dest_stats.columns = ['DestinationID_x', 'dest_avg_rating', 'dest_review_count', 
                              'dest_avg_children', 'dest_avg_popularity']
        df = df.merge(dest_stats, on='DestinationID_x', how='left')
        
        user_stats = df.groupby('UserID').agg({'Rating': ['mean', 'count']}).reset_index()
        user_stats.columns = ['UserID', 'user_avg_rating', 'user_review_count']
        df = df.merge(user_stats, on='UserID', how='left')
        
        state_stats = df.groupby('State').agg({'Rating': 'mean'}).reset_index()
        state_stats.columns = ['State', 'state_avg_rating']
        df = df.merge(state_stats, on='State', how='left')
        
        # Fit scaler
        numerical_features = ['NumberOfAdults', 'NumberOfChildren', 'total_travelers',
            'has_children', 'children_ratio', 'is_family', 'Popularity', 'dest_avg_rating',
            'dest_review_count', 'dest_avg_children', 'user_avg_rating', 'user_review_count',
            'state_avg_rating', 'is_adventure', 'is_cultural']
        self.feature_scaler.fit(df[numerical_features].values)
        
        return df
    
    def build_destination_cache(self, df):
        """Build destination feature cache"""
        return df.groupby('DestinationID_x').agg({
            'Name_x': 'first', 'State': 'first', 'Type': 'first', 'BestTimeToVisit': 'first',
            'dest_avg_rating': 'first', 'dest_review_count': 'first', 'Popularity': 'first',
            'dest_encoded': 'first', 'state_encoded': 'first', 'type_encoded': 'first',
            'time_encoded': 'first', 'dest_avg_children': 'first'
        }).reset_index()
    
    def get_recommendations(self, state, num_adults, num_children, preferences, best_time, destination_type, top_n):
        """Generate recommendations"""
        try:
            candidates = self.destination_features[
                self.destination_features['State'].str.lower() == state.lower()
            ].copy()
            
            if len(candidates) == 0:
                return None, f"No destinations found for: {state}"
            
            if destination_type and destination_type != "All Types":
                type_candidates = candidates[candidates['Type'].str.lower().str.contains(destination_type.lower(), na=False)]
                if len(type_candidates) > 0:
                    candidates = type_candidates
            
            total_travelers = num_adults + num_children
            has_children = 1 if num_children > 0 else 0
            children_ratio = num_children / (total_travelers + 1e-8)
            is_family = 1 if total_travelers >= 3 else 0
            
            pref_cleaned = preferences.lower().strip()
            pref_encoded = self.pref_encoder.transform([pref_cleaned])[0] if pref_cleaned in self.pref_encoder.classes_ else 0
            time_encoded = self.time_encoder.transform([best_time])[0] if best_time in self.time_encoder.classes_ else 0
            
            similar_mask = ((self.df_processed['has_children'] == has_children) &
                           (self.df_processed['total_travelers'].between(total_travelers-1, total_travelers+1)))
            similar_users = self.df_processed[similar_mask]['user_encoded'].unique()
            proxy_user = int(np.median(similar_users)) if len(similar_users) > 0 else int(self.df_processed['user_encoded'].median())
            
            n_candidates = len(candidates)
            user_avg_rating = self.df_processed[self.df_processed['user_encoded'].isin(similar_users)]['Rating'].mean() if len(similar_users) > 0 else self.df_processed['Rating'].mean()
            user_avg_count = self.df_processed[self.df_processed['user_encoded'].isin(similar_users)].groupby('user_encoded').size().mean() if len(similar_users) > 0 else 5
            state_avg_rating = self.df_processed[self.df_processed['State'].str.lower() == state.lower()]['Rating'].mean()
            
            is_adventure = 1 if 'adventure' in pref_cleaned or 'trek' in pref_cleaned else 0
            is_cultural = 1 if 'cultural' in pref_cleaned or 'heritage' in pref_cleaned else 0
            
            numerical_features = np.array([num_adults, num_children, total_travelers, has_children, children_ratio, is_family,
                candidates['Popularity'].mean(), candidates['dest_avg_rating'].mean(), candidates['dest_review_count'].mean(),
                num_children, user_avg_rating, user_avg_count, state_avg_rating, is_adventure, is_cultural]).reshape(1, -1)
            numerical_scaled = np.repeat(self.feature_scaler.transform(numerical_features), n_candidates, axis=0)
            
            X_pred = {
                'user_input': np.full(n_candidates, proxy_user),
                'dest_input': candidates['dest_encoded'].values,
                'state_input': candidates['state_encoded'].values,
                'type_input': candidates['type_encoded'].values,
                'time_input': np.full(n_candidates, time_encoded),
                'pref_input': np.full(n_candidates, pref_encoded),
                'numerical_input': numerical_scaled
            }
            
            predictions = self.model.predict(X_pred, verbose=0, batch_size=256).flatten()
            candidates = candidates.copy()
            candidates['predicted_score'] = predictions
            candidates['popularity_score'] = (candidates['predicted_score'] * 0.40 +
                candidates['dest_avg_rating'].fillna(0.5) * 0.30 +
                np.log1p(candidates['dest_review_count'].fillna(1)) * 0.15 +
                (candidates['Popularity'].fillna(50) / 100) * 0.15)
            
            if num_children > 0:
                family_boost = candidates['dest_avg_children'].fillna(0) > 0.5
                candidates.loc[family_boost, 'popularity_score'] *= 1.2
            
            time_match = candidates['BestTimeToVisit'].str.lower().str.contains(best_time.lower(), na=False)
            candidates.loc[time_match, 'popularity_score'] *= 1.1
            
            recommendations = candidates.nlargest(top_n, 'popularity_score')
            output = recommendations[['Name_x', 'State', 'Type', 'BestTimeToVisit', 'predicted_score',
                'dest_avg_rating', 'dest_review_count', 'Popularity', 'popularity_score']].copy()
            output.columns = ['Destination', 'State', 'Type', 'Best_Time', 'AI_Score', 'Avg_Rating', 'Reviews', 'Popularity', 'Final_Score']
            output['AI_Score'] = output['AI_Score'].round(3)
            output['Avg_Rating'] = output['Avg_Rating'].round(2)
            output['Final_Score'] = output['Final_Score'].round(3)
            
            return output, None
        except Exception as e:
            return None, f"Error generating recommendations: {str(e)}"


def create_recommendation_card(row, rank):
    """Create a beautiful recommendation card"""
    type_emojis = {'Historical': '🏛️', 'Nature': '🌿', 'Beach': '🏖️', 'Religious': '🕌',
                   'Adventure': '🏔️', 'Cultural': '🎭', 'Wildlife': '🦁', 'Hill Station': '⛰️'}
    emoji = type_emojis.get(row['Type'], '📍')
    score_color = "#11998e" if row['Final_Score'] >= 0.8 else "#f093fb" if row['Final_Score'] >= 0.6 else "#667eea"
    
    return f"""
    <div class="recommendation-card">
        <h2 style="color: #667eea; margin-bottom: 10px;">{emoji} #{rank} {row['Destination']}</h2>
        <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
            <span><strong>📍 State:</strong> {row['State']}</span>
            <span><strong>🏛️ Type:</strong> {row['Type']}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
            <span><strong>📅 Best Time:</strong> {row['Best_Time']}</span>
            <span><strong>⭐ Rating:</strong> {row['Avg_Rating']}/5.0</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span><strong>💬 Reviews:</strong> {int(row['Reviews'])}</span>
            <span><strong>🔥 Popularity:</strong> {int(row['Popularity'])}/100</span>
        </div>
        <div style="background: {score_color}; color: white; padding: 10px; border-radius: 8px; 
                    text-align: center; margin-top: 15px;">
            <strong>AI Match Score: {row['Final_Score']:.2f}/1.00</strong>
        </div>
    </div>
    """


def main():
    """Main Streamlit App"""
    
    st.markdown('<h1 class="main-title">🌍Tourism Recommender</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Discover your perfect destination with artificial intelligence</p>', unsafe_allow_html=True)
    
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.recommendations = None
    
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/around-the-globe.png", width=100)
        st.title("⚙️ Configuration")
        
        st.info("📂 Enter the full path to your files")
        
        model_path = st.text_input(
            "Model Path (.h5)", 
            value=r"D:\CDAC\Project\TravelDeep\best_model.h5",
            help="Full path to best_model.h5 file"
        )
        
        csv_path = st.text_input(
            "Data Path (.csv)", 
            value=r"D:\CDAC\Project\TravelDeep\final_df.csv",
            help="Full path to final_df (1).csv file"
        )
        
        if st.button("🚀 Initialize System", use_container_width=True):
            with st.spinner("Loading AI model and data..."):
                try:
                    app = TourismRecommenderApp(model_path, csv_path)
                    
                    # Load model
                    model, error = app.load_model()
                    if error:
                        st.error(f"❌ {error}")
                        st.stop()
                    app.model = model
                    
                    # Load data
                    df, error = app.load_data()
                    if error:
                        st.error(f"❌ {error}")
                        st.stop()
                    
                    # Process data
                    app.df_processed = app.fit_encoders(df)
                    app.destination_features = app.build_destination_cache(app.df_processed)
                    
                    st.session_state.app = app
                    st.session_state.initialized = True
                    st.success("✅ System initialized successfully!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Initialization failed: {str(e)}")
        
        st.markdown("---")
        if st.session_state.initialized:
            app = st.session_state.app
            st.markdown("### 📊 System Stats")
            st.metric("States", app.df_processed['State'].nunique())
            st.metric("Destinations", app.df_processed['DestinationID_x'].nunique())
            st.metric("Model Params", f"{app.model.count_params():,}")
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.info("""
        - Use full Windows paths with backslashes
        - Ensure both files exist
        - Model file should be .h5 format
        - CSV should match training format
        """)
    
    if not st.session_state.initialized:
        st.warning("👈 Please initialize the system from the sidebar!")
        
        col1, col2, col3 = st.columns(3)
        for col, icon, title, desc in zip(
            [col1, col2, col3],
            ['🎯', '⚡', '🌟'],
            ['Personalized', 'Fast & Accurate', 'Data-Driven'],
            ['AI learns your preferences', 'Instant predictions', 'Real traveler data']
        ):
            with col:
                st.markdown(f"""
                <div style="background: white; padding: 30px; border-radius: 15px; text-align: center;">
                    <h2>{icon}</h2><h3>{title}</h3><p>{desc}</p>
                </div>
                """, unsafe_allow_html=True)
        return
    
    app = st.session_state.app
    
    st.markdown("## 📝 Your Travel Preferences")
    col1, col2 = st.columns(2)
    
    with col1:
        state = st.selectbox("📍 Select State", sorted(app.df_processed['State'].unique()))
        num_adults = st.slider("👨 Adults", 1, 10, 2)
        preferences = st.selectbox("❤️ Interests", 
            ['General Tourism', 'Adventure', 'Beach', 'Cultural', 'Nature', 'Religious', 'Historical', 'Wildlife'])
    
    with col2:
        destination_type = st.selectbox("🏛️ Type", ['All Types'] + sorted(app.df_processed['Type'].unique()))
        num_children = st.slider("👶 Children", 0, 5, 0)
        best_time = st.selectbox("📅 Best Time", sorted(app.df_processed['BestTimeToVisit'].unique()))
    
    top_n = st.slider("🔢 Number of Recommendations", 3, 20, 10)
    
    if st.button("✨ Generate Recommendations", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(100):
            progress_bar.progress(i + 1)
            msg = "🔍 Analyzing..." if i < 30 else "🤖 AI Processing..." if i < 60 else "✨ Finding matches..."
            status_text.markdown(f'<p style="text-align:center; color:white; font-size:1.5rem;">{msg}</p>', unsafe_allow_html=True)
            time.sleep(0.02)
        
        progress_bar.empty()
        status_text.empty()
        
        recommendations, error = app.get_recommendations(
            state, num_adults, num_children, preferences, best_time,
            destination_type if destination_type != "All Types" else None, top_n
        )
        
        if error:
            st.error(f"❌ {error}")
        else:
            st.session_state.recommendations = recommendations
    
    if st.session_state.recommendations is not None:
        recommendations = st.session_state.recommendations
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                    color: white; padding: 20px; border-radius: 10px; text-align: center; 
                    font-size: 1.2rem; margin: 20px 0;">
            🎉 Found your perfect destinations!
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        for col, value, label in zip(
            [col1, col2, col3, col4],
            [len(recommendations), f"{recommendations['Final_Score'].max():.2f}", 
             f"{recommendations['Avg_Rating'].mean():.1f}/5", f"{int(recommendations['Reviews'].sum())}"],
            ['Destinations', 'Top Score', 'Avg Rating', 'Total Reviews']
        ):
            with col:
                st.markdown(f'<div class="metric-box"><h3>{value}</h3><p>{label}</p></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["📋 Recommendations", "📊 Scores", "🎯 Top Pick"])
        
        with tab1:
            for idx, row in recommendations.iterrows():
                st.markdown(create_recommendation_card(row, idx + 1), unsafe_allow_html=True)
        
        with tab2:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=recommendations['Destination'],
                y=recommendations['Final_Score'],
                marker=dict(color=recommendations['Final_Score'], colorscale='Viridis', showscale=True),
                text=recommendations['Final_Score'].round(3),
                textposition='auto'
            ))
            fig.update_layout(
                title="AI Match Scores Comparison",
                xaxis_title="Destination",
                yaxis_title="Score",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(recommendations, use_container_width=True)
        
        with tab3:
            top = recommendations.iloc[0]
            st.markdown(f"### 🏆 {top['Destination']}")
            st.markdown(f"**📍 Location:** {top['State']} | **🏛️ Type:** {top['Type']}")
            st.markdown(f"**📅 Best Time:** {top['Best_Time']} | **⭐ Rating:** {top['Avg_Rating']}/5.0")
            st.markdown(f"**💬 Reviews:** {int(top['Reviews'])} | **🔥 Popularity:** {int(top['Popularity'])}/100")
            st.markdown(f"### 🎯 AI Match Score: {top['Final_Score']:.2f}/1.00")
            
            categories = ['AI Score', 'Rating', 'Popularity', 'Reviews']
            values = [top['AI_Score']*100, (top['Avg_Rating']/5.0)*100, 
                     top['Popularity'], min(100, (top['Reviews']/100)*100)]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values, theta=categories, fill='toself',
                marker=dict(color='#667eea'), line=dict(color='#667eea')
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                title=f"Metrics for {top['Destination']}",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()