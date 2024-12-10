import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from streamlit_tags import st_tags

# Set page configuration
st.set_page_config(page_title='Food Court Menu Recommendation System', page_icon='🍕', layout='wide')

# Initialize session state for liked items, cart, and page navigation
if 'liked_items' not in st.session_state:
    st.session_state.liked_items = []

if 'cart' not in st.session_state:
    st.session_state.cart = []

if 'selected_letter' not in st.session_state:
    st.session_state.selected_letter = 'A'

if 'previous_page' not in st.session_state:
    st.session_state.previous_page = 'Home'

if 'page' not in st.session_state:
    st.session_state.page = 'Home'

@st.cache_data
def load_data():
    menu_data = pd.read_csv('updated_dataset.csv')
    menu_data['description'] = menu_data['description'].fillna('')
    menu_data['combined_text'] = (
        menu_data['category_x'] + ' ' + 
        menu_data['name_x'] + ' ' + 
        menu_data['description'] * 3
    )
    menu_data['original_rating'] = menu_data['Rating']
    menu_data['original_price'] = menu_data['price']
    scaler = StandardScaler()
    menu_data[['Rating', 'price']] = scaler.fit_transform(menu_data[['Rating', 'price']])
    return menu_data

@st.cache_resource
def prepare_model(menu_data):
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
    tfidf_matrix = tfidf.fit_transform(menu_data['combined_text'])
    numerical_features = menu_data[['Rating', 'price']]
    extended_numerical_features = csr_matrix(numerical_features.values)
    combined_features = hstack([tfidf_matrix, extended_numerical_features])
    svd = TruncatedSVD(n_components=150)
    reduced_features = svd.fit_transform(combined_features)
    return reduced_features

def cosine_similarity_manual(matrix):
    dot_product = np.dot(matrix, matrix.T)
    norm = np.linalg.norm(matrix, axis=1)
    similarity = dot_product / (norm[:, None] * norm[None, :])
    return similarity

@st.cache_resource
def calculate_similarity(reduced_features):
    return cosine_similarity_manual(reduced_features)

def recommend_menu(menu_name, menu_data, cosine_sim, top_n=10):
    if menu_name not in menu_data['name_x'].values:
        raise ValueError(f"{menu_name} not found in menu.")
    input_idx = menu_data[menu_data['name_x'] == menu_name].index[0]
    sim_scores = list(enumerate(cosine_sim[input_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_items = sim_scores[:top_n]
    recommended_menus = [{
        'Menu Item': menu_data.loc[i, 'name_x'],
        'Category': menu_data.loc[i, 'category_x'],
        'Rating': menu_data.loc[i, 'original_rating'],
        'Description': menu_data.loc[i, 'description'],
        'Price': menu_data.loc[i, 'original_price'],
        'Similarity Score': score,
        'Booth Number': menu_data.loc[i, 'restaurant_id'],  # Booth Number
        'Booth Name': menu_data.loc[i, 'name_y'],  # Booth Name
        'Image URL': menu_data.loc[i, 'url']  # Image URL
    } for i, score in top_items]
    return recommended_menus

def create_recommendation_card(rec, liked, in_cart):
    return f"""
    <div class="recommendation-card">
        <div class="card-header">
            <h2>{rec['Booth Name']}</h2>
            <h4>{rec['Menu Item']}</h4>
        </div>
        <div class="card-body">
            <img src="{rec['Image URL']}" alt="{rec['Menu Item']}" style="width:100%; height:auto; border-radius:10px; margin-bottom:10px;">
            <div class="booth-info">
                <p><strong>Booth Number:</strong> {rec['Booth Number']}</p>
                <p><strong>Category:</strong> {rec['Category']}</p>
                <p><strong>Rating:</strong> {rec['Rating']}</p>
                <p><strong>Price:</strong> ${rec['Price']:.2f}</p>
                <p><strong>Description:</strong> {rec['Description']}</p>
                <p><strong>Similarity Score:</strong> {rec['Similarity Score']:.2f}</p>
            </div>
        </div>
    </div>
    """


# Load data
menu_data = load_data()
reduced_features = prepare_model(menu_data)
cosine_sim = calculate_similarity(reduced_features)

# Streamlit UI
st.sidebar.title("🍙 RecSys 🍙")
st.sidebar.markdown("## Find Your Perfect Dish!")
page = st.sidebar.radio("Navigation", ["Home", "Browse Menu", "Liked Items", "Cart"])

# Apply custom CSS for background image and theme
st.markdown(
    f"""
    <style>
    body {{
        background-image: url('static/fc.jpg');
        background-size: 50%;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #333;
    }}
    .stApp {{
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 10px;
        margin: 20px;
    }}
    .recommendation-card {{
        background-color: #E2EFE2;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 10px;
        overflow: hidden;
        width: 100%;
    }}
    .card-header {{
        background-color: #EFE2EF;
        border-radius: 10px 10px 0 0;
        color: #fff;
        padding: 10px;
        text-align: center;
    }}
    .card-header h2 {{
        font-size: 20px;
        margin-bottom: 5px;
    }}
    .card-header h4 {{
        font-size: 18px;
        margin-bottom: 5px;
    }}
    .card-body {{
        padding: 15px;
    }}
    .booth-info {{
        color: #333;
        font-size: 14px;
        margin-bottom: 5px;
    }}
    .card-actions {{
        align-items: center;
        display: flex;
        justify-content: flex-end;
        margin-top: 10px;
    }}
    .like-button, .cart-button {{
        background-color: #fff;
        border: 1px solid #ccc;
        border-radius: 5px;
        cursor: pointer;
        margin-right: 5px;
        padding: 5px 10px;
    }}
    .like-button:hover, .cart-button:hover {{
        background-color: #f0f0f0;
    }}
    .info-message {{
    background-color: #f0f8ff;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #d0e3f0;
    margin-bottom: 20px;
}}
.info-message p {{
    margin: 5px 0;
    color: #333;
}}
.info-message strong {{
    color: #0056b3;
}}
    </style>
    """,
    unsafe_allow_html=True
)

# Home page with search and recommendations
if page == "Home":
    st.title("✨ Welcome to Our Food Court Menu Recommendation System! ✨")
    st.write("Discover delightful dishes tailored just for you. Enter keywords or select a dish to get started.")
    st.image('static/fc.jpg', use_column_width=True)
    st.markdown("---")

    # Add note explaining recommendation functionality
    st.markdown("""
    <div class="info-message">
        <p><strong>How This Works:</strong></p>
        <p>This page displays personalized recommendations based on the dish you select. The suggestions are tailored to the keywords entered based on dish name, dish category and description of the dish.</p>
        <p><strong>Note:</strong> You can use the search bar to find a dish and receive recommendations for similar items. The similarity score indicates how closely related the dishes are, ranging from 0 (not related at all) to 1 (most closely related).</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Autocomplete tags input
    keywords = st_tags(label='Enter keywords related to dishes:', text='Press enter to add more', suggestions=['pizza', 'burger', 'pasta'], maxtags=5, key='1')

    if keywords:
        keyword = ' '.join(keywords)
        filtered_data = menu_data[menu_data['combined_text'].str.contains(keyword, case=False, na=False)]
        if not filtered_data.empty:
            dish_name = st.selectbox('Select a dish to get recommendations:', filtered_data['name_x'].unique())
            if st.button('Get Recommendations'):
                recommendations = recommend_menu(dish_name, menu_data, cosine_sim)
                st.markdown("### Top Recommendations:")
                for i in range(0, len(recommendations), 5):
                    cols = st.columns(5)
                    for col, rec in zip(cols, recommendations[i:i+5]):
                        liked = rec['Menu Item'] in st.session_state.liked_items
                        in_cart = rec['Menu Item'] in st.session_state.cart
                        with col:
                            st.markdown(create_recommendation_card(rec, liked, in_cart), unsafe_allow_html=True)

# Browse Menu page
elif page == "Browse Menu":
    st.title("🍽️ Browse All Menu Items")
    st.write("Explore our full menu and find new dishes to try.")
    st.markdown("---")

    # Add note explaining browse functionality
    st.markdown("""
    <div class="info-message">
        <p><strong>How This Works:</strong></p>
        <p>This page allows you to explore our entire menu. Use the search bar and filters to find specific items or categories. You can view detailed information about each dish, including the booth it comes from, price, and rating.</p>
        <p><strong>Note:</strong> This section is designed for browsing and discovering new items. You can add items to your cart or save them to your favorites for quick access later.</p>
    </div>
    """, unsafe_allow_html=True)
    
    keyword_input = st.text_input('Enter keywords to filter dish names:', '')
    
    if keyword_input:
        price_range = st.slider('Select price range:', float(menu_data['original_price'].min()), float(menu_data['original_price'].max()), (float(menu_data['original_price'].min()), float(menu_data['original_price'].max())))
        rating_range = st.slider('Select rating range:', float(menu_data['original_rating'].min()), float(menu_data['original_rating'].max()), (float(menu_data['original_rating'].min()), float(menu_data['original_rating'].max())))
        
        filtered_data = menu_data.copy()
        if keyword_input:
            filtered_data = filtered_data[filtered_data['name_x'].str.contains(keyword_input, case=False)]
        filtered_data = filtered_data[
            (filtered_data['original_price'] >= price_range[0]) &
            (filtered_data['original_price'] <= price_range[1]) &
            (filtered_data['original_rating'] >= rating_range[0]) &
            (filtered_data['original_rating'] <= rating_range[1])
        ]
        
        st.write(f"### Found {len(filtered_data)} items matching the criteria.")
        
        if not filtered_data.empty:
            for idx, row in filtered_data.iterrows():
                liked = row['name_x'] in st.session_state.liked_items
                in_cart = row['name_x'] in st.session_state.cart
                st.markdown("""
                <style>
                .recommendation-card {
                    background-color: #E2EFE2;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                    width: 100%;
                    overflow: hidden;
                    position: relative; /* Ensure relative positioning for absolute button positioning */
                }
                .recommendation-card h3 {
                    color: #333;
                }
                .recommendation-card p {
                    margin-bottom: 10px;
                }
                </style>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="recommendation-card">
                    <h3>🍲 {row['name_x']}</h3>
                    <p><strong>Category:</strong> {row['category_x']}</p>
                    <p><strong>Rating:</strong> {row['original_rating']}</p>
                    <p><strong>Price:</strong> ${row['original_price']:.2f}</p>
                    <p><strong>Description:</strong> {row['description']}</p>
                </div>
                """, unsafe_allow_html=True)

                # Button actions
                if st.button(f"{'❤️' if liked else '🤍'} Like", key=f"like_{row['name_x']}"):
                    if row['name_x'] in st.session_state.liked_items:
                        st.session_state.liked_items.remove(row['name_x'])
                    else:
                        st.session_state.liked_items.append(row['name_x'])
                    st.experimental_rerun()

                if st.button(f"{'🛒' if in_cart else '➕'} Add to Cart", key=f"cart_{row['name_x']}"):
                    if row['name_x'] in st.session_state.cart:
                        st.session_state.cart.remove(row['name_x'])
                    else:
                        st.session_state.cart.append(row['name_x'])
                    st.experimental_rerun()

        else:
            st.write('No dishes found matching the criteria.')

# Liked Items page
elif page == "Liked Items":
    st.title("❤️ Liked Items")
    st.write("View and manage your favorite dishes here.")
    st.markdown("---")
    
    # Apply custom CSS for card styling
    st.markdown("""
    <style>
    .recommendation-card {
        background-color: #E2EFE2;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        width: 60%;
        overflow: hidden;
    }
    .recommendation-card h3 {
        color: #333;
        font-size: 20px;
    }
    .recommendation-card .booth-info {
        margin-bottom: 10px;
    }
    .recommendation-card .booth-info p {
        margin: 5px 0;
    }
    .recommendation-card p {
        margin-bottom: 10px;
        color: #555;
    }
    .card-actions {
        display: flex;
        justify-content: space-between;
        margin-top: 10px;
    }
    .card-actions button {
        background-color: #f5f5f5;
        border: 1px solid #ccc;
        border-radius: 5px;
        cursor: pointer;
        padding: 8px 16px;
    }
    .card-actions button:hover {
        background-color: #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if st.session_state.liked_items:
        for item in st.session_state.liked_items:
            item_data = menu_data[menu_data['name_x'] == item].iloc[0]
            st.markdown(f"""
            <div class="recommendation-card">
                <h3>{item_data['name_x']}</h3>
                <div class="booth-info">
                    <p><strong>Booth Number:</strong> {item_data['restaurant_id']}</p>
                    <p><strong>Booth Name:</strong> {item_data['name_y']}</p>
                </div>
                <p><strong>Category:</strong> {item_data['category_x']}</p>
                <p><strong>Rating:</strong> {item_data['original_rating']}</p>
                <p><strong>Price:</strong> ${item_data['original_price']:.2f}</p>
                <p><strong>Description:</strong> {item_data['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.write("You haven't liked any items yet.")

# Cart page
elif page == "Cart":
    st.title("🛒 Your Cart")
    st.write("Review the items you have added to your cart.")
    st.markdown("---")
    
    # Apply custom CSS for card styling
    st.markdown("""
    <style>
    .recommendation-card {
        background-color: #E2EFE2;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        width: 60%;
        overflow: hidden;
    }
    .recommendation-card h3 {
        color: #333;
        font-size: 20px;
    }
    .recommendation-card .booth-info {
        margin-bottom: 10px;
    }
    .recommendation-card .booth-info p {
        margin: 5px 0;
    }
    .recommendation-card p {
        margin-bottom: 10px;
        color: #555;
    }
    .card-actions {
        display: flex;
        justify-content: space-between;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if st.session_state.cart:
        for item in st.session_state.cart:
            item_data = menu_data[menu_data['name_x'] == item].iloc[0]
            st.markdown(f"""
            <div class="recommendation-card">
                <h3>{item_data['name_x']}</h3>
                <div class="booth-info">
                    <p><strong>Booth Number:</strong> {item_data['restaurant_id']}</p>
                    <p><strong>Booth Name:</strong> {item_data['name_y']}</p>
                </div>
                <p><strong>Category:</strong> {item_data['category_x']}</p>
                <p><strong>Rating:</strong> {item_data['original_rating']}</p>
                <p><strong>Price:</strong> ${item_data['original_price']:.2f}</p>
                <p><strong>Description:</strong> {item_data['description']}</p>
                
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.write("Your cart is empty.")











