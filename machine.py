import torch
import torch.nn as nn
from bertBaseModel import BaseModel
from main import Metrics

def supervised_kmeans_clustering(vectors, labels, dev_vectors, dev_labels, num_clusters, batch_size, epochs):
    # Perform k-means clustering with 5 clusters and the given labels
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)


    for epoch in range(epochs):
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            kmeans.fit(batch_vectors, batch_labels)
        # kmeans.fit(vectors, labels)

        # Output validation results after each epoch

        dev_predictions = kmeans.predict(dev_vectors)
        # print("prediction:", dev_predictions)
        # print("true:", dev_labels.numpy())
        metrix = Metrics(dev_labels, dev_predictions)
        acc = metrix.accuracy()
        precision = metrix.precision()
        recall = metrix.recall()
        f1 = metrix.f1()
        # accuracy = sum(dev_predictions == dev_labels.numpy()) / len(dev_labels.numpy())
        print(f"Epoch {epoch + 1} validation accuracy: {acc}, precision:{precision}, recall:{recall}, f1:{f1}")


# In[16]:


import numpy as np
from sklearn.svm import SVC
def svm_classification(train_samples, train_labels, dev_samples, dev_labels, num_classes, batch_size, epochs):
    # Initialize the SVM classifier with the given number of classes
    svm = SVC(kernel='linear', C=1, decision_function_shape='ovr', max_iter=1000, random_state=0)

    for epoch in range(epochs):
        for i in range(0, len(train_samples), batch_size):
            batch_samples = train_samples[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]
            svm.fit(batch_samples, batch_labels)

        # Output validation results after each epoch
        dev_predictions = svm.predict(dev_samples)
        metrix = Metrics(dev_labels, dev_predictions)
        acc = metrix.accuracy()
        precision = metrix.precision()
        recall = metrix.recall()
        f1 = metrix.f1()
        print(f"Epoch {epoch + 1} validation accuracy: {acc}, precision:{precision}, recall:{recall}, f1:{f1}")


# In[17]:


from sklearn.tree import DecisionTreeClassifier

def decision_tree_classification(train_samples, train_labels, dev_samples, dev_labels, num_classes, batch_size, epochs):
    # Initialize the decision tree classifier with the given number of classes
    decision_tree = DecisionTreeClassifier(max_depth=5, random_state=0)

    for epoch in range(epochs):
        for i in range(0, len(train_samples), batch_size):
            batch_samples = train_samples[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]
            decision_tree.fit(batch_samples, batch_labels)

        # Output validation results after each epoch
        dev_predictions = decision_tree.predict(dev_samples)
        metrix = Metrics(dev_labels, dev_predictions)
        acc = metrix.accuracy()
        precision = metrix.precision()
        recall = metrix.recall()
        f1 = metrix.f1()
        print(f"Epoch {epoch + 1} validation accuracy: {acc}, precision:{precision}, recall:{recall}, f1:{f1}")


# In[37]:


from sklearn.linear_model import LinearRegression

def linear_regression_prediction(train_samples, train_labels, dev_samples, dev_labels, num_classes, batch_size, epochs):
    # Initialize the linear regression model
    linear_regression = LinearRegression()

    for epoch in range(epochs):
        for i in range(0, len(train_samples), batch_size):
            batch_samples = train_samples[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]
            linear_regression.fit(batch_samples, batch_labels)

        # Output validation results after each epoch
        dev_predictions = linear_regression.predict(dev_samples)
        metrix = Metrics(dev_labels, dev_predictions)
        acc = metrix.accuracy()

        precision = metrix.precision()
        recall = metrix.recall()
        f1 = metrix.f1()
        print(f"Epoch {epoch + 1} validation accuracy: {acc}")

class Machine(BaseModel):
    def __init__(self, Shape,
                 args,
                 **kwargs):
        super(Machine, self).__init__(args.bert_dir, dropout_prob=args.dropout_prob)
        self.args = args

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                span1_mask=None,
                span2_mask=None):
        bert_outputs = self.bert_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        logits = []
        if self.args.model_type == '1d':
            # for bert_outputs_i in list(bert_outputs.hidden_states):
            #     token_out = bert_outputs_i # [batch, max_seq_len, dim]
            token_out = bert_outputs[0]  # [batch, max_seq_len, dim]
            seq_out = bert_outputs[1]  # [batch, dim]
            logit = []

            for t_out, s_out, s1_mask, s2_mask in zip(token_out, seq_out, span1_mask, span2_mask):
                s1_mask = s1_mask == 1
                s2_mask = s2_mask == 1
                span1_out = t_out[s1_mask]
                span2_out = t_out[s2_mask]
                # out = torch.cat([s_out.unsqueeze(0), span1_out, span2_out], dim=0).unsqueeze(0)
                # 这里可以使用最大池化或者平均池化，使用平均池化的时候要注意，
                # 要除以每一个句子本身的长度
                # out = torch.sum(out, 1)
                #
                out = torch.cat(
                    [torch.mean(span1_out, 0).unsqueeze(0), torch.mean(span2_out, 0).unsqueeze(0)],
                    dim=1)
                out = torch.where(torch.isnan(out), torch.full_like(out, 0), out)
                logit.append(out)
            logits.append(logit)
        elif self.args.model_type == '2d':
            for bert_outputs_i in list(bert_outputs.hidden_states):
                token_out = bert_outputs_i  # [batch, max_seq_len, dim]
                # token_out = bert_outputs[0]  # [batch, max_seq_len, dim]
                seq_out = bert_outputs[1]  # [batch, dim]
                logit = []

                for t_out, s_out, s1_mask, s2_mask in zip(token_out, seq_out, span1_mask, span2_mask):
                    s1_mask = s1_mask == 1
                    s2_mask = s2_mask == 1
                    span1_out = t_out[s1_mask]
                    span2_out = t_out[s2_mask]
                    # out = torch.cat([s_out.unsqueeze(0), span1_out, span2_out], dim=0).unsqueeze(0)
                    # 这里可以使用最大池化或者平均池化，使用平均池化的时候要注意，
                    # 要除以每一个句子本身的长度
                    # out = torch.sum(out, 1)
                    #
                    out = torch.cat(
                        [torch.mean(span1_out, 0).unsqueeze(0), torch.mean(span2_out, 0).unsqueeze(0)],
                        dim=1)
                    out = torch.where(torch.isnan(out), torch.full_like(out, 0), out)
                    logit.append(out)
                logits.append(logit)
        for i in range(len(logits)):
            logits[i] = torch.stack([torch.tensor(x).clone() for x in logits[i]], dim=0)
        x = torch.nn.functional.normalize(
            torch.stack([torch.tensor(x).clone() for x in logits], dim=0).squeeze().transpose(0, 1), p=2, dim=1)
        if self.args.model_type == '1d':
            x = x.transpose(0, 1).unsqueeze(1)

        return x.squeeze(1)
