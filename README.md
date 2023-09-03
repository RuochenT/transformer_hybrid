# Transformer-based Hybrid Recommender System

## 1. Introduction 
This study aims to investigate the effectiveness of three Transformers (BERT, RoBERTa, XLNet) in handling data sparsity and cold start problems in the recommender system. We present a Transformer-based hybrid recommender system that predicts missing ratings and ex- tracts semantic embeddings from user reviews to mitigate the issues. We conducted two experi- ments on the Amazon Beauty product review dataset, focusing on multi-label text classification under sparse data and recommendation performance in user cold start settings. Our findings demonstrate that XLNet outperforms the other Transformers in both tasks, and our proposed methods show superior performance compared to the traditional ALS method in handling data sparsity and cold start scenarios. This study not only confirms transformers’ effectiveness under cold start conditions in recommender systems but also highlights the need for further study and improvement in the fine-tuning process.

## 2. Data 
The dataset is originally from the Amazon Review Data in 2018 (Ni et al., 2019). We used the Beauty category of the Amazon datasets, which are widely regarded as benchmarks in research papers. The dataset consists of 371,345 rows and 12 columns, with 324,038 unique user IDs and 32,586 unique item IDs. Moreover, the average number of words in the review text is 64 words.

### Data Descriptive Statistics

Our goal is to investigate whether our proposed method can effectively handle situations where there are limited historical user ratings and scarce data. Therefore, the relevant variables are ratings, user ID, item ID, and review text. Figure 3.1 shows that the rating variable is highly imbalanced among users, with more than 60% of users giving a rating of 5. Figure 3.2 presents a bar chart displaying the number of users and their ratings, where more than 99% of users in the dataset have rated less than 10 times.
Moreover, the distribution of beauty products in Figure 3 illustrates that most items have been rated by only a few users or have received few ratings. The analysis between userID and ItemID suggests that the cold start setting is likely to appear in the data set due to limited historical user ratings.
Figure 3.3 shows a word cloud comparison between users who rated more than 3 (green) and users who rated less than 3 (orange). From the word cloud, it can be observed that words such as ”love,” ”skin,” ”hair,” ”body,” and ”perfect” appear frequently when users rate more than 3. On the other hand, words like ”money,” ”shave,” ”bad,” and ”waste” seem to occur more often when users rate less than 3.

![Figure_1](https://github.com/RuochenT/transformer_hybrid/assets/119982930/2e809974-7abb-453e-9429-004a64cfffbd)
![Figure_3](https://github.com/RuochenT/transformer_hybrid/assets/119982930/6317f668-a491-464f-9bfa-ab27c8468610)
![word_cloud](https://github.com/RuochenT/transformer_hybrid/assets/119982930/a0593dbb-870b-4bef-8ced-bb8badf26758)

### Data preprocessing

The duplicated rows and the missing values of review text were dropped to get more accurate recommendations and facilitate the deep learning part using transformers. Moreover, we did one- hot encoding on the rating variable to 5 numerical variables (1-5). Consequently, users who rated more than 20 ratings were removed to construct cold start settings (Feng et al., 2021). The first experiment focuses on rating classification based on the reviews from transformers. As a result, the data is split randomly with a 70:15:15 approach into training, test, and validation data sets.
17
We excluded ratings randomly for 30% of users in the test set to depict a more challenging cold start scenario for the first experiment. Moreover, dummy variables were created for the rating variable, which is a necessary step for multi-label classifications in transformer models.
The second experiment aims to investigate whether sentence embeddings from transformers can enhance recommender systems when there are new users entered the system. As a result, we split the data based on unique user IDs to make sure that users in the training set and test set are not overlapped. Table 3.1 shows two final data sets for the second experiment with unique users, unique items, and the number of ratings.


## 3. Methodology

We developed the method to alleviate cold start problem by leveraging transformers to classify the missing rating and encode the review into contextual sentence embeddings as a side information for the recommender system. The method should be effective when there is missing rating data and limited user reviews. We incorporated both rating and embeddings into collective matrix factorization to get the recommendations for new users. We conducted two experiments to evaluate the effectiveness of the proposed method. Figure 4.4 depicts the visualization of our proposed method.

<img width="412" alt="Screenshot 2566-09-03 at 17 37 09" src="https://github.com/RuochenT/transformer_hybrid/assets/119982930/678ea306-1fd9-4efb-97f5-f41c79050236">

## 4. Results 
### the multi-label text classification 
#### The model learning curves 
We initially set the number of epochs to 4 at the beginning of the training process, following the recommendation in the paper. However, we monitored the behavior of each model by analyzing the training loss and validation loss. Eventually, we decided to select the models at epoch 2. Figure 5.1, 5.2, and 5.3 provides valuable insights as they show that the training losses were consistently higher than the validation losses and gradually decreased with each epoch, indicating the models were learning. However, for all three models, the validation loss started to increase after the second epoch. This increase indicated that the models were overfitting to the training data and struggling to generalize effectively to the validation data set beyond 2 epochs.

![BERT](https://github.com/RuochenT/transformer_hybrid/assets/119982930/dcd5630a-40c4-4c73-af75-c69db45b5bc4)
![RoBERTa](https://github.com/RuochenT/transformer_hybrid/assets/119982930/a8a5da7c-c587-4229-b212-fb8c8b1108a3)
![XLNet](https://github.com/RuochenT/transformer_hybrid/assets/119982930/866b6a67-7a18-4004-90be-787eb72b41d1)

#### The Model Performance
It is important to note that we evaluated based on the missing 30% rating rather than the entire rating column. There was not much difference between BERT and RoBERTa performances across three metrics. BERT yielded an accuracy of 84% for accuracy which was slightly higher than the accuracy of RoBERTa at 83.6%. Both models achieved a weighted F1 of 0.830. However, BERT obtained a slightly higher AUC-ROC score at 0.746 compared to RoBERTa’s score of 0.746. On the other hand, XLNet outperformed both models across the three metrics. It achieved an accuracy of 84.9%, which is 1.1% and 1.6% higher than BERT and RoBERTa, respectively. Moreover, it obtained a weighted F1 score of 0.840, which is 1.2% higher than BERT and RoBERTa. In terms of AUC-ROC scores, it achieved 0.750, which was slightly higher than BERT and RoBERTa by 0.54% and 0.40%, respectively.

| Models | Accuracy | Weighted F1 |AUC-ROC |
| --- | --- | --- | --- |
| BERT |0.840 | 0.830 |0.747| 
| RoBERTa| 0.836| 0.830|0.746| 
| XLNet | **0.849**| **0.840**|**0.750** | 

### Recommender system
In the second experiment, we evaluated the performance of our proposed method in compar- ison to the baseline models which are Collective Matrix Factorization with BERT, RoBERTa, and XLNet sentence embeddings, as well as Alternating Least Squares (ALS) with and without missing values datasets. The complete rating results from XLNet were used, and the data was split into non-overlapping users into training and test sets to simulate cold start scenarios for users. The detailed results are presented in Table 5.2.

| Methods | Prec@3 | Prec@5 | Recall@3| Recall@5 |MAP@3 |MAP@5|
| --- | --- | --- | --- | --- | --- | --- | 
| ALS |0.289| 0.261 |0.260 |0.259 |0.289|0.261
| ALS_XL|0.883| 0.876 |0.782 |0.870 |0.880|0.869|0.859
| BERT based|0.885 |0.878 |0.784 |0.872 |0.872|0.862|
| Roberta based |0.886 |0.878 |0.784 |0.872 |0.872|0.862|
| XLNet based |**0.887**| **0.880** |**0.784**|**0.872** |**0.872**|**0.862**|

## 5. Discussion 
From our first experiment, we made two main discoveries. Firstly, the performance of RoBERTa and BERT was comparable, showing minimal differences. Secondly, XLNet out- performed BERT and RoBERTa in the multi-label classification task. In the multi-label text classification task, we expected that RoBERTa will outperform BERT based on the paper by Liu et al. (2019). However, the performance of RoBERTa was similar to that of BERT based on accuracy, weighted F1, and AUC-ROC metrics. It seems that the removal of NSP (Next Sentence Prediction) and the implementation of dynamic masking during the pre-training pro- cess may not have had a substantial impact on multi-label text classification. Furthermore, it is important to note that Liu et al. (2019) primarily evaluated RoBERTa on tasks related to semantic similarity and question answering, which may explain the limited benefits observed in our specific task of multilabel text classification. On the other hand, XLNet outperformed two models by no more than 1.6% across three metrics. It demonstrates that the XLNet architecture may be more effective in this specific dataset and the task. However, XLNet was trained on more diverse and larger datasets than BERT and RoBERTa during pre-training (Yang et al., 2019). Therefore, we cannot conclude that XLNet’s architecture is superior in this specific fine-tuning task.

In our second experiment, we identified three key findings. Firstly, incorporating complete ratings improved the performance of the traditional recommender system significantly (ALS and ALS XL). Secondly, systems utilizing transformers outperformed baseline models, showcasing the overall enhancement achieved through transformer-based approaches. Lastly, while there were minor variations, the results from different transformers were relatively similar, with XL- Net demonstrating a slight advantage. In the recommender system experiment, Alternating Least Square (ALS) with the original data set provided the worst results as expected. This incomplete rating information introduces data sparsity problems in the recommender system. As a result, the algorithm cannot capture the complete user-item interactions and lead to un- reliable factorization results (Jannach et al., 2010). Moreover, it is even more challenging for
37
the algorithm to predict the rating for unknown users in the test dataset. On the other hand, we anticipated that the ALS technique with the complete rating dataset (ALS XL) would also provide significantly inferior results compared to our proposed method. In theory, matrix fac- torization struggles to generate accurate recommendations in cold start settings (Jannach et al., 2010; Roy & Dutta, 2022). Surprisingly, it was able to generate compatible performances in terms of precision, recall, and mean Average Precision (mAP). One of the possible explanations is that ALS generates recommendations for new users by identifying overlapping items from similar users in the training data sets based on the latent representations between existing users and new users. However, our proposed models still generated better results overall.

Another important point is that the performances of the three transformers are similar across all six metrics, with the XLNet-based system performing slightly better in precision. One possible explanation for this similarity in performance is the positive bias and limited information present in the reviews. Based on figure 3.1, the reviews in the data set mostly contain positive feedback (more than 60% rated 5 for items). Three transformers may capture similar overall sentiment information and result in similar overall sentence embeddings and lead to similar recommender performance. Moreover, figure 3.3 shows that the reviews predominantly contain limited set of words such as ”love,” ”skin,” and ”hair” for positive feedback, and ”money,” ”shave,” ”bad,” and ”waste” for negative feedback. As a result, these reviews may not sufficiently represent the unique preferences of each customer and have a narrow set of topics. As a result, this limits the ability of the models to capture substantial variations between users. However, the XLNet-based system still demonstrated better performances in handling cold-start scenarios and providing recommendations than the ALS technique. Our proposed method effectively addressed the challenges of data sparsity and cold start problems, outperforming the traditional ALS method.

## 6. Limitations 
There are four main limitations in our study. First, we evaluated based on a small sample of 10,000 observations due to the computational limitations and the long training time. In the data preprocessing step, there were 324,033 rows left after filtering out users with more than 20 interactions and removing duplicate rows. Consequently, it took us more than 10 hours to fine-tune each model. As a result, we decided to sample the data for 10,000 rows based on rating, which allows us to train each model within five hours. However, a larger data set can improve the learning curves and mitigate overfitting of Transformers. Second, we evaluated our proposed method with one type of dataset. It would improve the certainty and generalizability of our performance if we included multiple datasets. Third, our study compares the three specific transformers (BERT, RoBERTa, and XLNet). As a result, it would improve the study to be more comprehensive to compare them with other text transformers, such as Electra and Albert. Fourth, we cannot conclude that XLNet is the best model based on the results. This is because the model was trained on more than ten times larger and more diverse data sets (Yang et al., 2019). XLNet should be built from scratch and trained on the same datasets as BERT for a fair comparison. However, due to time and resource constraints, we were unable to improve this. Nevertheless, our proposed method sheds light on the benefits of combining transformers and hybrid recommender systems in cold start settings, as well as their limitations.
