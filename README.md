# Neural Collaborative Filtering Book Recommendation System 
# and 
# Combined Guided Topic Modeling-Based and Aspect-Based Sentiment Analysis For Book User Reviews
Combined repository for my Master's practicum 1 and 2 projects: Neural Collaborative Filtering Book Recommendation System and Combined Guided Topic Modeling-Based and Aspect-Based Sentiment Analysis For Book User Reviews. Both projects use the same data.


# Neural Collaborative Filtering Book Recommendation System

## Note
Due to file sizes, the notebooks had to be split up and the data could not be uploaded.

## Introduction
Recommendation systems generate lists of items a user might be interested in. A well-known example of this is movie recommendation in streaming content providers like Netflix or Hulu. Collaborative filtering recommendation systems suggest items for a user based on what similar users are interested in. In this case, it is book ratings. This can be accomplished without knowing anything about the item's content such as genre, actor, or author. Those are used in content filtering recommendation systems and is out of scope for this project. It is useful that you can make recommendations for a user based on other users. However, when an item has no ratings then there is nothing to link it between users. This is known as the cold start problem.
  
![collab filtering diagram](https://github.com/DarrellS0352/msds692_s40_data_science_practicum_1/blob/fca1428cb1223c21aceb25b46d3ce43d7d928362/images/collaborative%20filtering%20diagram.png)

Neural collaborative filtering (NCF) combines neural networks and collaborative filtering. It essentially comes down to user ID, item ID, and the user rating. It takes the user ID's and item ID's as inputs, feeds them through matrix factorization, and then through a Multi-Layer Perceptron Network (MLP). Mapping the user-item ratings along the way. It is possible to include more inputs in the equation such as what users click on.

![ncf diagram](https://github.com/DarrellS0352/msds692_s40_data_science_practicum_1/blob/4995b3984cb928918e2fa4605f2d15b558a08d2b/images/ncf%20diagram.png)

## The Data

The data used is a subset of Amazon review data and comes from two files: review data and book metadata. The primary review data contains all types of items found on Amazon, but only the book data was used. It consisted of 51,311,621 reviews. The book metadata contained information on 2,935,525 unique books. The compressed file sizes were 11.81 gb review data and 1.22 gb metadata. There were numerous variables in each, but only the book ID, book title, and book category (genre) were imported from the book metadata file while the user ID, book ID, user rating, and review text was imported in the reviews file to reduce computation and storage resource usage.

## Data Ingestion and Wrangling

The project was completed using Python 3.8 in Jupyter Notebooks on a laptop. The laptop ended up being a bottleneck. With limited storage and computing resources, alternate methods were devised to avoid crashing the IDE when importing raw data. Standard methods like a Pandas import were insufficient so custom functions using the were designed to import data. These functions used the gzip, json, and os libraries to read through the compressed files, pull the desired fields from each row, and load them into a dictionary. For development purposes, row count limiting functionality was also built into it.

```
def load_amazon_review_data(file_name, nrows = 100000000):
    counter = 0
    data = []
    # loading compressed json file
    with gzip.open(file_name) as fin:
        for l in fin:
            d = json.loads(l)
            # only verified reviews
            if d['verified'] == True:
                # only pulling necessary fields to reduce memory allocation
                d2 = {}
                d2['user_id'] = d.get('reviewerID')
                d2['book_id'] = d.get('asin')
                d2['rating'] = d.get('overall')
                d2['review_text'] = d.get('reviewText')
                data.append(d2)
                counter += 1
            # nrows break
            if (nrows is not None) and (counter > nrows):
                break
    return data
# load data
amazon_reviews = load_amazon_review_data(os.path.join(input_filepath, 'AmazonBooks.json.gz'))
```

The raw data could be loaded into a Pandas Dataframe following the initial import.

![raw data merged DF](https://github.com/DarrellS0352/msds692_s40_data_science_practicum_1/blob/7209d956b9924ae53c81a6ff2bb5f579b1579aaf/images/raw%20data%20merged%20DF.PNG)

After some data exploration and testing, it became apparent the data was still too large for the available resources. Therefore, filtering by book genre and ratings count was implemented. I ended up taking the top ten highest frequency book genres and only using books that had fifty or more ratings. Some additional wrangling was done to the Literature & Fiction genre because it had sub-genres that were the same as other major genres (ex: mystery thriller under literature & fiction being moved the mystery, thriller, & suspense genre). The remaining literature & fiction sub-genres were labeled as genre fiction.

![genre freqs](https://github.com/DarrellS0352/msds692_s40_data_science_practicum_1/blob/2e861b7bea6ecc18347fd28da5c5a8872060a6ab/images/genre%20freqs.PNG)

## Exploratory Data Analysis

Given the data consisted only of user ID's, book ID's, book titles, and user ratings, an in-depth EDA was not necessary. ID values are unique and have no conceptual value. There is the review text but that was earmarked for some extra text analysis functionality which was de-scoped from the project. I've noticed when personally browsing books, that books with popular movies tend to have the most ratings the data appears to support the theory (ex: The Martian, The Hunger Games, and the Lord of the Rings.)

![most reviewed books](https://github.com/DarrellS0352/msds692_s40_data_science_practicum_1/blob/8622cdb71a2e851c03c7edd33ab0185601d7da88/images/EDA%20most%20reviewed%20books.PNG)

It also seems most people who take the time to review something tend to give positive reviews. It may be a systemic flaw for recommendation systems if people do not share their negative feedback which would improve recommendation performance.

![rating distribution](https://github.com/DarrellS0352/msds692_s40_data_science_practicum_1/blob/8622cdb71a2e851c03c7edd33ab0185601d7da88/images/EDA%20rating%20distribution.PNG)

## Modeling

The model architecture was simple. It takes the book ID input and user ID input, creates embeddings, and runs them through a few neural network dense layers using dropout for regularization.

```
output_dim = 10 # larger number = higher capacity model prone to overfitting - extends training time

# book embedding layer
input_books = Input(shape=[1], name='book_input')
embed_books = Embedding(book_n + 1, output_dim, name='book_embed')(input_books)
embed_books_output = Flatten(name='book_embed_output')(embed_books)

# user embedding layer
input_users = Input(shape=[1], name='user_input')
embed_users = Embedding(user_n + 1, output_dim, name='user_embed')(input_users)
embed_users_output = Flatten(name='user_embed_output')(embed_users)

concat = Concatenate()([embed_books_output, embed_users_output])
concat_dropout = Dropout(0.2)(concat)
dense1 = Dense(128, activation='relu')(concat_dropout)
dense1_dropout = Dropout(0.2)(dense1)
dense2 = Dense(64, activation='relu')(dense1_dropout)
dense2_dropout = Dropout(0.2)(dense2)

output = Dense(1, name='output', activation='relu')(dense2_dropout)

model = Model(inputs=[input_books, input_users], outputs=output)

optimizer = Adam()

model.compile(
    optimizer=optimizer, 
    loss='mean_squared_error'
)
```
![model architecture](https://github.com/DarrellS0352/msds692_s40_data_science_practicum_1/blob/bab6fcefbfedf7fd2f1529ad9b4f803c807647dd/images/model.png)

Model training times were excessive due to the data size and processing resources. Training data was over 13 million rows after the prior filtering completed in the data wrangling stage. Running on CPU would take 20-30 hours to run. I attempted to move to the cloud using Google Colab but they rapidly throttled my account and cut me off. Therefore, I setup my laptop to train the neural network using the GPU. This made training much faster and if you setup the batch size to fit the GPU (64, 128, 256, 512, etc.), it is even more efficient.

## Model Outputs

Given the unsupervised nature of the effort, three different dimensionality reduction methods were used to plot and assess the embedding weight outputs of the model. Principle Component Analysis (PCA), T-Distributed Stochastic Neighbor Embedding (TSNE), and Uniform Manifold Approximation and Projection (UMAP). All were used to reduce the outputs to two dimensions so they can be viewed using scatter plots. The book genres were represented with colors.

![PCA plot](https://github.com/DarrellS0352/msds692_s40_data_science_practicum_1/blob/b2b5e3e15808da9a9b1cb50808d9fa60b738c795/images/pca%20output%20all%20genres.PNG)

![UMAP plot](https://github.com/DarrellS0352/msds692_s40_data_science_practicum_1/blob/b2b5e3e15808da9a9b1cb50808d9fa60b738c795/images/umap%20output%20all%20genres.PNG)

![TSNE plot](https://github.com/DarrellS0352/msds692_s40_data_science_practicum_1/blob/b2b5e3e15808da9a9b1cb50808d9fa60b738c795/images/tsne%20output%20all%20genres.PNG)

There was a lot of overlap in the fiction genres (mystery, fiction, science fiction and fantasy). It was interesting to find Christian books and bibles, children's books, and the self-help type books were overlapping each other but largely removed from the fiction genres. The below plots have a single book genre in black with the remaining genres in red. If you compare the “mystery, thriller, & suspense” plot against the “Christian books & bibles” plot, they seem to be negatives of each other fitting like puzzle pieces.

![TSNE mystery](https://github.com/DarrellS0352/msds692_s40_data_science_practicum_1/blob/b2b5e3e15808da9a9b1cb50808d9fa60b738c795/images/tsne%20output%20mystery.PNG)

![TSNE bibles](https://github.com/DarrellS0352/msds692_s40_data_science_practicum_1/blob/b2b5e3e15808da9a9b1cb50808d9fa60b738c795/images/tsne%20output%20bibles.PNG)

### Recommendations

Making recommendations is as simple as making predictions for any other model. You just predict for a user. The recommendation function limited the recommendations to five books in the output.

```
def book_rec(user_id):
    book_id = list(reviews.book_id.unique())
    book_array = np.array(book_id)
    user_array = np.array([user_id for i in range(len(book_id))])
    prediction = model.predict([book_array, user_array])
    prediction = prediction.reshape(-1)
    prediction_ids = np.argsort(-prediction)[0:5]
    
    recommended_books = books.loc[books.book_id.isin(prediction_ids)]
    return recommended_books
 ```
 
 After making predictions, you can see how the user's previously rated books visually relate to the recommended books.
 
 ![PCA recommendations](https://github.com/DarrellS0352/msds692_s40_data_science_practicum_1/blob/9f9d1dfe9fbd0c8de7b35b80592d1bcbfba83dc4/images/Recs%20PCA.PNG)
 
 ![TSNE recommendations](https://github.com/DarrellS0352/msds692_s40_data_science_practicum_1/blob/9f9d1dfe9fbd0c8de7b35b80592d1bcbfba83dc4/images/Recs%20TSNE.PNG)
 
 ![UMAP recommendations](https://github.com/DarrellS0352/msds692_s40_data_science_practicum_1/blob/9f9d1dfe9fbd0c8de7b35b80592d1bcbfba83dc4/images/Recs%20umap.PNG)
 
# Combined Guided Topic Modeling-Based and Aspect-Based Sentiment Analysis For Book User Reviews

## Objective
The purpose of this project is to give additional insights about recommended books to a user utilizing topic modeling, aspect extraction, and sentiment analysis. It is intended to be an extension of a previous project that provides book recommendations with neural networks which can be found at [this repository](https://github.com/DarrellS0352/Neural-Collaborative-Filtering-Book-Recommendation-System).

<p align="center">
 <img src="https://github.com/DarrellS0352/Combined-Guided-Topic-Modeling-Based-and-Aspect-Based-Sentiment-Analysis-For-Book-User-Reviews/blob/main/img/topic%20sentiment%20review%20example.png" width=35% height=35%> <img src="https://raw.githubusercontent.com/DarrellS0352/Combined-Guided-Topic-Modeling-Based-and-Aspect-Based-Sentiment-Analysis-For-Book-User-Reviews/main/img/process%20flow.PNG" width=100% height=100%>
 </p>
 
## The Data
The data used is Amazon product review data gathered by Ni et al. (2019). There is raw review data, ratings only data, data subsets for products that have at least five reviews, and per-category data available for each product category. The raw data has book metadata (2,935,525 books) and the book review data (51,311,621 reviews). The two datasets are joined on the book ID to add book titles to the reviews. To meet computing constraints, the data is then filtered to the top 10 genres with the most unique books and books that have 50 or more reviews. The final dataset is approximately 13 million reviews for 64,000 books and 5.3 million users. For development and presentation considerations, the data was filtered to a single book with 1,539 reviews.

<p align="center">
<img src="https://raw.githubusercontent.com/DarrellS0352/Combined-Guided-Topic-Modeling-Based-and-Aspect-Based-Sentiment-Analysis-For-Book-User-Reviews/main/img/genre%20freqs.PNG" width=35% height=35%>
 </p>

### Data Pre-processing
The raw review texts were pre-processed with the following methods:
+ Unicode removal
+ HTML removal
+ URL removal
+ Emoticon removal
+ Expanded contractions (they're > they are)
+ Forced lowercase
+ Split reviews into individual sentences
## Aspect Extraction
Aspect extraction was done using KeyBERT and some custom wrangling methods. [KeyBERT](https://maartengr.github.io/KeyBERT/index.html) utilizes BERT embeddings to produce keywords/phrases that best represent a document. It is used in combination with KeyphraseCountVectorizer which is an enhanced version of the CountVectorizer and is designed to find key phrases using part-of-speech patterns and also takes care of stopwords. The part-of-speech patterns used are variants from the work of Banjar et al. (2020). It parses syntactic dependencies pulling out phrases such as adjective > noun or verb > noun. The model is fed a list of seed keywords which guide the model (if possible) towards phrases related to said list. The seed keywords are various words related to aspects of writing and books. The words used are setting, character, plot, conflict, theme, point-of-view, tone, style, dialogue, action, description, exposition, motivation, symbolism, climax, resolution, imagery, pacing, writing, and author. The output is matched lists of nouns and their descriptive words.

<p align="center">
<img src="https://raw.githubusercontent.com/DarrellS0352/Combined-Guided-Topic-Modeling-Based-and-Aspect-Based-Sentiment-Analysis-For-Book-User-Reviews/main/img/aspect%20DF.PNG" width=25% height=25%>
 </p>

## Guided Topic Modeling
A [sentence transformer embedding model](https://huggingface.co/sentence-transformers) is used to create numerical representations of the cleaned sentences. The sentence embeddings are then given to the topic model. The topic modeling is done with [BERTopic](https://maartengr.github.io/BERTopic/index.html). The guided topic modeling functionality is utilized by also feeding the list of seed keywords to this model. This determines the high-level topics written about in the reviews using transformers and class-based term frequency inverse document frequency (TF-IDF) to create dense topic clusters. Each topic produced consists of the top four keywords of the topic concatenated with underscores (ex: windows_drive_dos_file). The topics are assigned a topic number with -1 being outliers that don't fit with others well enough.

<p align="center">
<img src="https://raw.githubusercontent.com/DarrellS0352/Combined-Guided-Topic-Modeling-Based-and-Aspect-Based-Sentiment-Analysis-For-Book-User-Reviews/main/img/topic%20DF.PNG" width=60% height=60%>
 </p>

## Sentiment Analysis
The pre-trained sentiment classification model used was the [siebert/sentiment-roberta-large-english sentiment model](https://huggingface.co/siebert/sentiment-roberta-large-english?text=I+like+you.+I+love+you). The sentiment model was applied to the key phrase for aspect-based sentiment analysis. For the topic modeling-based sentiment analysis it was applied to the cleaned sentence which is represented by the assigned topic.
 
### Aspect-Based Sentiment Analysis
Aspect-based sentiment analysis extracts aspects (nouns) and their descriptors (adjectives, verbs, etc.) and applies sentiments to the extracted words at a sub-sentence level. Most sentiment analyses produce a macro-level sentiment for a single word, sentence, paragraph, document, etc. In this case, it is the sentiments in book reviews for the elements of writing and books from the seed keyword list.

The sentiment model gives 0-1 scores with POSITIVE or NEGATIVE labels. The scores with NEGATIVE labels are then reversed in polarity and all nouns are lemmatized. Given the output includes every noun that met the part-of-speech patterns, the output is filtered to just the words from the seed keyword list. Then the output is aggregated to determine the mean sentiment score and count of mentions for each word. The final product is a bar graph with the color indicating the mean sentiment and the x-axis portraying the aspect counts. The elements of writing and books provided by the seed keywords give decent metrics that can be used to evaluate most books. This plot can inform a potentional consumer regarding meta-information on how others liked the writing/author and also gives an idea of the weight of those sentiments (bar length/color saturation).

<p align="center">
<img src="https://raw.githubusercontent.com/DarrellS0352/Combined-Guided-Topic-Modeling-Based-and-Aspect-Based-Sentiment-Analysis-For-Book-User-Reviews/main/img/aspect%20sentiment%20plot.PNG" width=100% height=100%>
 </p>

### Guided Topic Modeling-Based Sentiment Analysis
The guided topic modeling-based sentiment analysis provides meta and content focused sentiments extracted from the book reviews. It is possible the model finds topics that are similar to the seed keyword aspects but it also produces book content-based topics. For example, the topic 'scifi_recommend_asimov_readable' gives the impression the book is comparable to Isaac Asimov novels. While the topic 'holden_miller_detective_idealistic' is describing two primary characters and someone unfamiliar with the story may not completely understand. The same wrangling techniques were applied except the original cleaned sentences were fed through the sentiment model which is acceptable because the sentences are represented by their assigned topics. The final product is also a bar graph reprsenting the mean sentiment of the topic and the x-axis portrays the topic counts.

<p align="center">
<img src="https://raw.githubusercontent.com/DarrellS0352/Combined-Guided-Topic-Modeling-Based-and-Aspect-Based-Sentiment-Analysis-For-Book-User-Reviews/main/img/topic%20sentiment%20plot.PNG" width=100% height=100%>
 </p>

## References
Banjar, A., Ahmed, Z., Daud, A., Abbasi R.A., Dawood, H. (2020). Aspect-Based Sentiment Analysis for Polarity Estimation of Customer Reviews on Twitter. Computers, Materials & Continua 2021, 67(2), 2203-2225. https://doi.org/10.32604/cmc.2021.014226 

Ni, J., Li, J., McAuley, J. (2019). Justifying Recommendations using Distantly-Labeled Reviews and Fine-Grained Aspects. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 188–197. http://dx.doi.org/10.18653/v1/D19-1018
