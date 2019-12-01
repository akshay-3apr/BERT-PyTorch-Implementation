<h1> BERT: <i><b>B</b>idirectional <b>E</b>ncoder <b>R</b>epresentations from <b>T</b>ransformers</i></h1>
<h6><i>created by Google AI Language Team in 2018</i></h6>
BERT is designed to pre-train deep bidirectional representations from unlabelled text by jointly conditioning on both left and right context in all the layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.
BERT is trained on unlabelled dataset to achieve state of the art results on 11 individual NLP tasks. And all of this with little fine tuning.

Deeply Bidirectional means that BERT learns information from both the left and right side of a token's context during the training.
<p>Let's try to understand the concept of left and right context in Deeply Bidirectional</p>
<ul>
<li>Sentence 1: They exchanged addresses <b>and agreed to keep in touch.</b></li>
<li>Sentence 2: <b>People of India will be</b> addressed by Prime Minister today.</li>
</ul>

If model is trained unidirectional and we try to predict the word <i><b>"Address"</b></i> from the above two sentences dataset, then the model will be making error in predicting either of them.

<h3> Word Embedding</h3>

![](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/06/06062705/Word-Vectors.png)

Before BERT, NLP community used features based on searching the key terms in the word corpus using Term Frequency.These vectors were used in mathematical and statistical models for classification and regression tasks. There was nothing much that could be done mathematically on term frequency to understand the syntax and semantics of the word in a sentence. Then arrived an era of word embedding. Here every word can be represented in their vector space and words having same meaning were close to each other in vector space. This started from Word2Vec and GloVe. 

Consider an example:
<ul>
<li>Sentence 1: Man is related to Woman</b></li>
<li>Sentence 2: Then King is related to ...</li>
</ul>

Above sentence can be explained mathematically as: <b>King - Man + Woman = Queen</b>

And this can be achieved using word embeddings.Only issue with such word embeddings was with respect to the information they could store. Word2Vec could store only feedforward information. Resulting in same vectors for similar words used in different context. Such words are know as <b>Polysemy</b> words. To handle polysemy words, prediction led to more complex and deeper LSTM models.

The revolutionary NLP architecture, which marked the era of transfer learning in NLP and also letting the model understand the syntax and semantics of a word, ELMo (<i>Embeddings from Language Models</i>) and ULMFit started the new trend. ELMo was then, the answer to the problem of <b>Polysemy</b> words- <i> same words having different meanings based on the context </i>.

<h2>Previous NLP model Architectures </h2>

![alt text](https://1.bp.blogspot.com/-RLAbr6kPNUo/W9is5FwUXmI/AAAAAAAADeU/5y9466Zoyoc96vqLjbruLK8i_t8qEdHnQCLcBGAs/s640/image3.png)

<i>BERT is deeply bidirectional, OpenAI GPT is unidirectional, and ELMo is shallowly bidirectional.</i>

<b>ELMo</b> used weighted sum of forward (<i>context before the token/word</i>) and backward (<i>context after the token/word</i>) pass generated, Intermediate Word vectors from two stacked biLM layers and raw vector generated from character convolutions to produce the final ELMo vector. This helped ELMo look at the past and future context, basically the whole sentence to generate the word vector, resulting in unique vector for Polysemy words.

The true power of transfer learning in NLP was unleashed after <b>ULMFiT</b> (<i>Universal Language Model Fine-tuning</i>). The concept revolved around having an Language Model (LM) trained on generic corpora. These LMs were based on same ideology what ImageNet helped to acheive transfer learning in Computer Vision. The stages in transfer learnng <b>pretraining</b> and <b>Fine-tuning</b> which is still followed now started with ULMFiT. In pretraining stage the LMs will be trained to learn generic information over language corpora. When fine-tuning the pretrained model to a downstream task, we will train the model on task specific data. Only the last few layers are the ones that will be trained from scratch. Resulting in better accurracy as the initial layers had generic language understanding and last layers had task specific information. BERT is based on the same idea that fine-tuning a pre-trained language model can help the model achieve better results in the downstream tasks.

Following ELMo and UMLFiT on the same ground, came <b>OpenAI GPT</b>(<i>Generative Pre-trained Transformers</i>). OpenAI GPT was based on Transformer based network, as suggested in Google Brains research
paper "[Attention is all you need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)". They replaced the whole LSTM architecture with encoder decoder layer stack. GPT also emphasized the importance of the Transformer framework, which has a simpler architecture and can train faster than an LSTM-based model. It is also able to learn complex patterns in the data by using the Attention mechanism. This started the breaktrough for NLP <i>state of the art</i> frameworks using <b>Transformers</b> which includes BERT.


<h2> Coming back to BERT... </h2>
BERT surpass the unidirectionality constraints by using a “<i>Masked Language Model (MLM)</i>” pre-training objective. MLM randomly masks some of the tokens from the input, and the objective is to predict the original vocabulary id of the masked word based only on its context. It enables the representation to fuse the left and the right context, which allows us to pretrain a deep bidirectional Transformer. In addition to the MLM, BERT also uses a “<i> next sequence prediction</i>” task that jointly pretrains text-pair representations.

There are two steps involved in BERT:

![](https://www.researchgate.net/profile/Jan_Christian_Blaise_Cruz/publication/334160936/figure/fig1/AS:776030256111617@1562031439583/Overall-BERT-pretraining-and-finetuning-framework-Note-that-the-same-architecture-in.ppm)


*   Pre-training: the model is trained on unlabelled data over different pre-training task.
*   Fine-tuning: BERT model is first initialized with the pre-trained parameters, and all of the parameters are fine-tuned using labelled data from the downstream task.

With the basic understanding of the above two steps, lets deep dive to understand BERT framework.


*   <h3>BERT Model Architecture:</h3>
BERT Model architecture is a multi-layer bidirectional Transformer encoder-decoder structure.
    
![](https://d3i71xaburhd42.cloudfront.net/0d2df885be9a4a8fe5cd9725d333c33ce6771057/2-Figure1-1.png)

   *   <b>Encoder</b>: Encoder is composed of a stack of N=6 identical layers. Each layer has two sub layers. The first layer is a multi-head self-attention mechanism, and the second is a position wise fully connected feed-forward network. There is a residual connection around each of the two sub layers, followed by layer normalization.

   *   <b>Decoder</b>: Decoder is also composed of N=6 identical layers. Decoder has additional one sub-layer over two sub-layers as present in encoder, which performs multi-head attention over the output of the encoder stack. Similar to encoder we have residual connection around every sub-layers, followed by layer normalization.

   *   <b>Attention</b>: Attention is a mechanism to know which word in the context, better contribute to the current word. It is calculated using the dot product between query vector Q and key vector K. The output from attention head is the weighted sum of value vector V, where the weights assigned to each value is computed by a compatibility function of the Query with the corresponding Key.
The general formula that sums up the whole process of attention calculation in BERT is:

   ![alt text](https://miro.medium.com/proxy/1*V6LGUR-0NmlOGmm0TDAa5g.png)

   where, Q is the matrix of queries, K an V matrix represent keys and values.

   To fully understand the attention calculation with example, I would request you to go through the [Analytics Vidya blog](https://www.analyticsvidhya.com/blog/2019/06/understanding-transformers-nlp-state-of-the-art-models/?utm_source=blog&utm_medium=demystifying-bert-groundbreaking-nlp-framework)

2.   <h3> Pre-training BERT:<h3> BERT is pretrained using two unsupervised task:
        <ul>
        <li> <b>Masked Language Model</b>: In order to train the bidirectional representation, BERT simply mask 15% of the input tokens at random, and then predict those masked tokens. A downside is that it creates a mismatch between pre-training and fine-tuning, since the [MASK] token does not appear during fine-tuning. To deal with this situation, BERT not always replaces the masked words with actual [MASKED] token. The BERT training data generator chooses 15% of the token positions at random for prediction. If the i-th token is chosen, BERT replaces the i-th token with: <ul><li> the [MASK] token 80% of the time</li><li>a random token 10% of the time</li><li>the unchanged i-th token 10% of the time</li></ul>
        </li>
        <li><b> Next Sentence Prediction (NSP)</b>: In order to train a model that understands sentence relationships, we pre-train for a next sentence prediction task. If there are two sentences A and B, BERT trains on 50% of the time with B as the actual next sentence that follows A (labeled as isNext), and 50% of the time it is a random sentence from the corpus (labeled as NotNext).
        </li>
        </ul>
3.   <h3>Fine-tuning BERT:</h3>The self-attention mechanism in the Transformer allows BERT to model any downstream task. BERT with self-attention encodes a concatenated text pair, which effectively includes bidirectional cross attention between two sentences. For each task, we simply plug in the task specific inputs and outputs into BERT and fine-tune all the parameters end to end. At the output the token representations are fed into an output layer for token level tasks, such as sequence tagging or question answering, and the [CLS] representation is fed into an output layer for classification, such as sentimental analysis or entailment.

<h2> Now, Lets start with BERT implementaion using PyTorch: </h2>
