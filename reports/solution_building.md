# Evaluation Metrics
The performance of method was evaluated using average similarity and average change of toxification of the sentence.

# Hypothesis 1: t5-small
I firstly tried to use t5-small on the whole dataset with three epochs. But when I started to use it, I found it not working well. For example, when I provided such sentence as an input: "Fuck you", it answers me like this: "fuck you fuck you fuck you fuck you". And there were many such problems.

# Hypothesis 2: t5-large

Then I decided to use t5-large, but each time I run this model I got an error: CUDA out of memory. Even with small batch size. 

# Hypothesis 3: The part of dataset for t5-base
Firstly, I decided not to use the whole dataset for the t5-base (I used only 20% of the dataset), because the time to wait was too long. In this case, the result was satisfying, it didn't show me strange messages as t5-small. Then I decided to try the whole dataset. 

# Results
I used the whole dataset for t5-base, but the result hasn't changed comparing to the previous case with 20% of dataset, 
