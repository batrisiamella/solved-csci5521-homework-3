Download Link: https://assignmentchef.com/product/solved-csci5521-homework-3
<br>
<ol>

 <li>Let {(<strong>x</strong><sub>1</sub><em>,y</em><sub>1</sub>)<em>,…,</em>(<strong>x</strong><em><sub>n</sub>,y<sub>n</sub></em>)} be a given training set where <strong>x</strong><em><sub>i </sub></em>∈ R<em><sup>d</sup>,y<sub>i </sub></em>∈ {0<em>,</em>1}. We consider the following regularized logistic regression objective function:</li>

</ol>

<em> ,</em>

where <em>λ &gt; </em>0 is a constant. Let <strong>w</strong><sup>∗ </sup>be the global minimizer of the objective, and let k<strong>w</strong><sup>∗</sup>k<sub>2 </sub>≤ <em>c</em>, for some known constant <em>c &gt; </em>0.

<ul>

 <li>Clearly show and explain the steps of the projected gradient descent algorithm for optimizing the regularized logistic regression objective function. The steps should include an exact expression for the gradient.</li>

 <li>Is the objective function strongly convex? Clearly explain your answer by stating and using the definition of strong convexity.<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a></li>

 <li>Is the objective function smooth? Clearly explain your answer by stating and using the definition of smoothness.<sup>1</sup></li>

 <li>Let <strong>w</strong><em><sub>T </sub></em>be the iterate after <em>T </em>steps of the projected gradient descent algorithm. What is a bound on the difference <em>f</em>(<strong>w</strong><em><sub>T</sub></em>)−<em>f</em>(<strong>w</strong><sup>∗</sup>)? Clearly explain all quantities in the bound (hint:please consider step size).</li>

</ul>

<ol start="2">

 <li>Let X = {<strong>x</strong><sub>1</sub><em>,…,</em><strong>x</strong><em><sub>n</sub></em>}<em>,</em><strong>x</strong><em><sub>i </sub></em>∈ R<em><sup>d </sup></em>be a set of <em>n </em>samples drawn i.i.d. from a mixture of <em>k </em>multivariate Gaussian distributions in R<em><sup>d</sup></em>. For component <em>G<sub>h</sub>,h </em>= 1<em>,…,k</em>, let <em>π<sub>h</sub>,µ<sub>h</sub>,</em>Σ<em><sub>h </sub></em>respectively denote the prior probability, mean, and covariance matrix of <em>G<sub>h</sub></em>. We will focus on the expectation maximization (EM) algorithm for learning the mixture model, in particular for estimating the parameters {(<em>π<sub>h</sub>,µ<sub>h</sub>,</em>Σ<em><sub>h</sub></em>)<em>,h </em>= 1<em>,…,k</em>} as well as the posterior probabilities <em>p</em>(<em>G<sub>h</sub></em>|<strong>x</strong><em><sub>i</sub></em>).

  <ul>

   <li>In your own words, describe the EM algorithm for mixture of Gaussians, highlighting the two key steps (E- and M-), illustrating the methods used in the steps on a high level, and what information they need.</li>

   <li>Assuming the posterior probabilities <em>p</em>(<em>G<sub>h</sub></em>|<em>x<sub>i</sub></em>) are known, show the estimates of the component prior, mean, and covariance <em>π<sub>h</sub>,µ<sub>h</sub>,</em>Σ<em><sub>h</sub>,h </em>= 1<em>,…,k </em>given by the Mstep (you do not need to show the derivation).</li>

   <li>Assuming the component prior, mean, and covariance <em>π<sub>h</sub>,µ<sub>h</sub>,</em>Σ<em><sub>h</sub>,h </em>= 1<em>,…,k </em>are known, show how the posterior probabilities <em>p</em>(<em>G<sub>h</sub></em>|<em>x<sub>i</sub></em>) are computed in the E-step.</li>

  </ul></li>

</ol>

<strong>Programming assignments:</strong>

The next two problems involve programming. For Question 3, we will be using the 2-class classification datasets from Boston50 and Boston75, and for Question 4, we will be using the the Digits dataset. For Q3, we will develop code for 2-class logistic regression with only one set of parameters (<strong>w</strong><em>,w</em><sub>0</sub>). For Q4, we will develop code for PCA.

<ol start="3">

 <li>We will develop code for 2-class logistic regression with one set of parameters (<strong>w</strong><em>,w</em><sub>0</sub>) where <strong>w </strong>∈ R<em><sup>d</sup>, w</em><sub>0 </sub>∈ R. Assuming the two classes are {0<em>,</em>1}, and the data <strong>x </strong>∈ R<em><sup>d</sup></em>, the posterior probability of class <em>C</em><sub>1 </sub>is given by</li>

</ol>

<em>T       </em>+ <em>w</em>0)

<em> ,</em>

1 + exp(<strong>w x </strong>+ <em>w</em><sub>0</sub>)

and <em>P</em>(0|<strong>x</strong>) = 1 − <em>P</em>(1|<strong>x</strong>). f We will develop code for MyLogisticReg2 with corresponding

MyLogisticReg2.fit(X,y) and MyLogisticReg2.predict(X) functions. Parameters for the model can be initialized following suggestions in the textbook. In the fit function, the parameters will be estimated using gradient descent as described in the textbook and in class.

We will compare the performance of MyLogisticReg2 with LogisticRegression<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a> on two datasets: Boston50 and Boston75. Using my cross val with 5-fold cross-validation, report the error rates in each fold as well as the mean and standard deviation of error rates across all folds for the two methods: MyLogisticReg2 and LogisticRegression, applied to the two 2-class classification datasets: Boston50 and Boston75.

You will have to submit (a) <strong>code </strong>and (b) <strong>summary of results</strong>:

(a) <strong>Code</strong>: You will have to submit code for MyLogisticReg2() as well as a wrapper code q3().

For MyLogisticReg2(), you are encouraged to consult the code for MultiGaussClassify() from HW2. You need to make sure you have init , fit, and predict implemented in MyLogisticReg2. init (d) will initialize the parameters (<strong>w</strong><em>,w</em><sub>0</sub>) and will take the data dimensionality <em>d </em>as input. fit(X,y) will take the data features <em>X </em>and labels <em>y </em>and will use gradient descent to estimate the parameters <strong>w</strong><em>,w</em><sub>0</sub>. Convergence of gradient descent can be determined by checking the difference between subsequent iterates

) and () and making sure the change is below a pre-specified (small)

threshold. You can also specify <em>t</em><sub>max</sub>, the maximum number of iterations gradient descent can run, and set it to a suitably large value. predict(X) will take a feature matrix corresponding to the test set and return the predicted labels. Your class MyLogisticReg2() will <strong>not </strong>inherit any base class in sklearn.

<strong>The wrapper code </strong>(main file q3.py) has no input and is used to prepare the datasets, and make calls to my cross val(method,<em>X</em>,<strong>y</strong>,<em>k</em>) to generate the error rate results for each dataset and each method. The code for my cross val(method,<em>X</em>,<strong>y</strong>,<em>k</em>) must be yours (e.g., code you wrote in HW1 with modifications as needed) and you cannot use cross val score() in sklearn. The results should be printed to terminal (not generating an additional file in the folder). Make sure the calls to my cross val(method,<em>X</em>,<strong>y</strong>,<em>k</em>) are made in the following order and add a print to the terminal before each call to show which method and dataset is being used:

<ul>

 <li>MyLogisticReg2 with Boston50;</li>

 <li>MyLogisticReg2 with Boston75;</li>

 <li>LogisticRegression with Boston50;</li>

 <li>LogisticRegression with Boston75.</li>

</ul>

One should be able to run your wrapper code q3.py by “python q3.py” in command line window.

(b) <strong>Summary of results</strong>: For each dataset and each method, report the test set error rates for each of the <em>k </em>= 5 folds, the mean error rate over the <em>k </em>folds, and the standard deviation of the error rates over the <em>k </em>folds. Make a table to present the results for each method and each dataset (4 tables in total). Each column of the table represents a fold and add two columns at the end to show the overall mean error rate and standard deviation over the <em>k </em>folds. For example:

<table width="365">

 <tbody>

  <tr>

   <td colspan="7" width="365">Error rates for MyLogisticReg2 with Boston50</td>

  </tr>

  <tr>

   <td width="56">Fold 1</td>

   <td width="56">Fold 2</td>

   <td width="56">Fold 3</td>

   <td width="56">Fold 4</td>

   <td width="56">Fold 5</td>

   <td width="51">Mean</td>

   <td width="35">SD</td>

  </tr>

  <tr>

   <td width="56">#</td>

   <td width="56">#</td>

   <td width="56">#</td>

   <td width="56">#</td>

   <td width="56">#</td>

   <td width="51">#</td>

   <td width="35">#</td>

  </tr>

 </tbody>

</table>

<ol start="4">

 <li>We will modify the provided ipython notebook generate digits.ipynb to add our own code for the parts where PCA is getting used. The notebook shows how to learn and subsequently generate data which look like the data from the Digits dataset. Let the data matrix be denoted as <em>X </em>∈ R<em><sup>D</sup></em><sup>×<em>n </em></sup>where <em>D </em>is the number of features and <em>n </em>is the number of samples. Each column of <em>X </em>corresponds to a data point <strong>x</strong><em><sub>j </sub></em>∈ R<em><sup>D</sup></em>. For Digits, we have <em>D </em>= 64 and <em>n </em>≈ 1800. The feature covariance matrix of the data can be computed as:</li>

</ol>

<em> ,                                   </em>(1)

where <em>µ</em>ˆ  is the mean of the data points. The notebook has the following steps:

<ul>

 <li>Compute a PCA projection <em>Z </em>∈ R<em><sup>d</sup></em><sup>×<em>n</em></sup><em>,d </em>≤ <em>D </em>of the original data <em>X </em>∈ R<em><sup>D</sup></em><sup>×<em>n </em></sup>so that <em>α</em>% of the variance is preserved in the projected space. The notebook has details for <em>α </em>= 90, where <em>d </em>= 21 was sufficient. The notebook also has details for <em>α </em>= 99, where <em>d </em>= 41 was sufficient. The projection can be represented by a matrix <em>W </em>∈ R<em><sup>d</sup></em><sup>×<em>D</em></sup>.</li>

</ul>

In the homework, instead of using sklearn’s functionality for determining <em>d </em>and <em>W</em>, we will develop <u>our own function </u>myPCA (see below) using numpy and scipy as needed. This will be the only major change you will have to do in the notebook.

<ul>

 <li>Consider the dataset Z = {<strong>z</strong><sub>1</sub><em>,…,</em><strong>z</strong><em><sub>n</sub></em>} of <em>n </em>points in R<em><sup>d</sup></em>, and fit a mixture of Gaussians (MoG) model where the number of mixture components is determined by a suitable model selection technique. The notebook already implements this part, and we will not change it.</li>

 <li>Sample a set of <em>m </em>new points Z˜ = {<strong>z</strong>˜<sub>1</sub><em>,…,</em><strong>z</strong>˜<em><sub>m</sub></em>}<em>,z</em>˜<em><sub>i </sub></em>∈ R<em><sup>d </sup></em>from the MoG model in the projected space. Let <em>Z</em>˜ ∈ R<em><sup>d</sup></em><sup>×<em>m </em></sup>be the matrix corresponding to the sampled data Z˜, where the columns are <strong>z</strong>˜<em><sub>i</sub></em>. The notebook already implements this part with <em>m </em>= 100, and we will not change it.</li>

 <li>Inverse transform the new points <em>Z</em>˜ in the <em>d</em>-dimensional space to the original <em>D</em>dimensional space. The notebook currently uses sklearn’s functionality for doing the inverse transformation. Instead we will use <u>our own code </u>to do the inverser transformation. In particular, the inverse transformation can be computed using <em>W </em>and <em>µ</em>ˆ which we will get as outputs from myPCA (see below). The inverse transform is a simple linear algebraic operation (say, 1-2 lines of code), so we will not write a separate function for this, but just inline the code.</li>

</ul>

The main function we will write is myPCA(<em>X</em>,<em>α</em>), which takes in the original data <em>X </em>∈ R<em><sup>D</sup></em><sup>×<em>n </em></sup>and the percentage <em>α </em>∈ [0<em>,</em>100] of variance to be preserved, and returns three things as a tuple:

<ul>

 <li>the PCA projection dimensionality <em>d </em>which is sufficient to preserve <em>α</em>% of the original variance in the projected space,</li>

 <li>the projection matrix <em>W </em>∈ R<em><sup>d</sup></em><sup>×<em>D</em></sup>, and</li>

 <li>the estimated mean ˆ<em>µ </em>∈ R<em><sup>D </sup></em>of the original data.</li>

</ul>

Add myPCA as a function in the notebook and update the rest of the notebook to use myPCA instead of sklearn’s PCA. Please make sure the current functionality in the notebook is maintained. In particular, your notebook should not import PCA but still maintain the same functionality. You will have to submit your notebook my generate digits.ipynb.

<a href="#_ftnref1" name="_ftn1">[1]</a> Since the function is twice differentiable, you can answer the question by considering the second derivative.

<a href="#_ftnref2" name="_ftn2">[2]</a> You should use LogisticRegression from scikit-learn, similar to HW1 and HW2.