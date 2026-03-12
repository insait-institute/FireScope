import re
import torch
from sentence_transformers import SentenceTransformer, util


'''
from sentence_transformers import CrossEncoder

# Cross-encoder trained for semantic textual similarity (scores ≈ [0, 1])
similarity_model = CrossEncoder('cross-encoder/stsb-roberta-large')
def similarity_reward(completions, references, **kwargs):
    batch_size=32
    completions = [c[0]['content'] for c in completions]
    # 1. Encode all candidates at once
    pairs = []
    owners = []  # which completion each pair belongs to

    # Build all (completion, reference) pairs
    for i, refs in enumerate(references):
        for r in refs:
            pairs.append((completions[i], r))
            owners.append(i)

    # Predict scores for all pairs (vectorized & batched inside)
    pair_scores = similarity_model.predict(pairs, batch_size=batch_size)

    # Aggregate: max score per completion
    out = [float("-inf")] * len(completions)
    for idx, s in zip(owners, pair_scores):
        if s > out[idx]:
            out[idx] = float(s)

    # If a completion had zero refs, return NaN
    return out
'''
similarity_model = SentenceTransformer('all-mpnet-base-v2')
def similarity_reward(completions, references, **kwargs):
    completions = [c[0]['content'] for c in completions]
    # 1. Encode all candidates at once
    cand_embs = similarity_model.encode(completions, convert_to_tensor=True)

    # 2. Flatten all references, record groupings
    flat_references = []
    ref_group_indices = []  # stores, for each reference, its candidate index
    for idx, refs in enumerate(references):
        flat_references.extend(refs)
        ref_group_indices.extend([idx]*len(refs))

    # 3. Encode all references at once
    ref_embs = similarity_model.encode(flat_references, convert_to_tensor=True)

    # 4. For each candidate, get their similarities
    scores = []
    ref_group_indices = torch.tensor(ref_group_indices)
    for i in range(len(completions)):
        idxs = (ref_group_indices == i).nonzero(as_tuple=True)[0]
        sims = util.cos_sim(cand_embs[i], ref_embs[idxs])  # shape: [1, num_refs_for_this_candidate]
        max_sim = sims.max().item()
        scores.append(max_sim)
    return scores

def is_final_answer_format_reward(completions, **kwargs):
    rewards = []
    completions = [c[0]['content'] for c in completions]
    for c in completions:
        try:
            x = int(c.split('FINAL ANSWER:\n')[-1])
            if x < 0 or x > 9:
                raise Exception("Invalid answer")
            rewards.append(1.0)
        except:
            rewards.append(0.0)
    return rewards

def generate_accuracy_reward(frequencies):
    def accuracy_reward(completions, solution, **kwargs):
        completions = [c[0]['content'] for c in completions]
        label = solution
        guesses = []
        mi = min(frequencies.values())
        ma = max(frequencies.values())
        frac_diff = ma/mi - 1.0
        weights = {x : 1 + frac_diff - frac_diff*(y - mi)/(ma-mi) for x, y in frequencies.items()}
        for c in completions:
            try:
                guess = int(c.split('FINAL ANSWER:\n')[-1].strip())
            except:
                guess = -100
            guesses.append(guess)
        
        # Classification reward
        cls_rewards = []
        for g, l in zip(guesses, label):
            if g == l:
                rew = 1
            elif abs(g-l) == 1:
                rew = 0.5
            elif abs(g-l) == 2:
                rew = 0.1
            else:
                rew = 0.0  # default fallback
            cls_rewards.append(rew*weights[l])

        return cls_rewards
    return accuracy_reward

def absolute_error(completions, solution, **kwargs):
    completions = [c[0]['content'] for c in completions]
    labels = solution
    guesses = []
    for c in completions:
        try:
            guess = int(c.split('FINAL ANSWER:\n')[-1].strip())
        except:
            guess = -100
        guesses.append(guess)
    
    # Classification reward
    absolute_errors = []
    for g, l in zip(guesses, labels):
        if g > 9 or g < 0:
            absolute_errors.append(9)
        else:
            absolute_errors.append(abs(g-l))

    return absolute_errors

def squared_error(completions, solution, **kwargs):
    completions = [c[0]['content'] for c in completions]
    labels = solution
    guesses = []
    for c in completions:
        try:
            guess = int(c.split('FINAL ANSWER:\n')[-1].strip())
        except:
            guess = -100
        guesses.append(guess)
    
    # Classification reward
    squared_errors = []
    for g, l in zip(guesses, labels):
        if g > 9 or g < 0:
            squared_errors.append(81)
        else:
            squared_errors.append((g-l)**2)

    return squared_errors

# fictive rewards used for reporting..
def final_answer(completions, **kwargs):
    completions = [c[0]['content'] for c in completions]
    answers = []
    for c in completions:
        try:
            a = int(c.split("FINAL ANSWER:\n")[-1])
        except:
            a = -1
        answers.append(a)
    return answers

def ground_truth(solution, **kwargs):
    return solution