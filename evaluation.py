from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def computeSimilarity(originalTextList, paraphraseTextList):
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    #model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')

    outputScores = []
    originalSentEmbeddings = model.encode(originalTextList)
    paraphraseSentEmbeddings = model.encode(paraphraseTextList)
    for originalSent, paraphraseSent in zip(originalSentEmbeddings, paraphraseSentEmbeddings):
        outputScores.append(cosine_similarity([originalSent], [paraphraseSent]).tolist()[0][0])

    return outputScores


def computeBLEU(originalTextList, paraphraseTextList):
    outputScores = []
    for originalSent, paraphraseSent in zip(originalTextList, paraphraseTextList):
        originalTokens = originalSent.split(' ')
        paraphraseTokens = paraphraseSent.split(' ')
        outputScores.append(sentence_bleu([originalTokens], paraphraseTokens, weights=(1, 0, 0, 0)))

    return outputScores


def computeROUGE(originalTextList, paraphraseTextList):
    outputScores = []
    for originalSent, paraphraseSent in zip(originalTextList, paraphraseTextList):
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        scores = scorer.score(originalSent, paraphraseSent)
        outputScores.append(scores['rougeL'][0])

    return outputScores


def completeEvaluate(originalTextFilename, generatedTextFilename, reportFilename, repeatTimes=1, split=False):
    originalList = []
    paraphraseList = []
    with open(originalTextFilename, 'r') as fr:
        for line in fr:
            for j in range(repeatTimes):
                originalList.append(line.strip())
    with open(originalTextFilename + '.listed', 'w') as fo:
        for data in originalList:
            fo.write(data + '\n')
    with open(generatedTextFilename, 'r') as fr:
        for line in fr:
            if split:
                paraphraseList += line.strip().split('\t')
            else:
                paraphraseList.append(line.strip())
    if split:
        with open(generatedTextFilename + '.listed', 'w') as fo:
            for data in paraphraseList:
                fo.write(data + '\n')

    print(len(originalList), len(paraphraseList))

    BLEUScores = computeBLEU(originalList, paraphraseList)
    ROUGEScores = computeROUGE(originalList, paraphraseList)
    print('Average BLEU: ' + str(sum(BLEUScores) / len(BLEUScores)))
    print('Average ROUGE: ' + str(sum(ROUGEScores) / len(ROUGEScores)))

    with open(reportFilename, 'w') as scoreReportFile:
        for bScore, rScore in zip(BLEUScores, ROUGEScores):
            scoreReportFile.write(str(bScore) + '\t' + str(rScore) + '\n')

    print('DONE')


def verifyKeyComponents(generatedFilename, itemFilename, reportFilename, repeatTimes=5):
    itemData = []
    with open(itemFilename, 'r') as fr:
        for line in fr:
            for j in range(repeatTimes):
                itemData.append(json.loads(line.strip())['NNPTK'])

    with open(itemFilename + '.list', 'w') as fo:
        for item in itemData:
            fo.write(item + '\n')

    reportFile = open(reportFilename, 'w')
    count = 0
    with open(generatedFilename, 'r') as fr:
        for lineIndex, line in enumerate(fr):
            if itemData[lineIndex].lower() in line.lower():
                reportFile.write('True\n')
                count += 1
            else:
                reportFile.write('False\n')
    reportFile.close()
    print(len(itemData))
    print(lineIndex)
    print(count)



if __name__ == '__main__':
    completeEvaluate('drive/My Drive/Cui_workspace/Data/TweetParaphrase/commTweets/commTweets.NNP.tokenized.sampled.original',
                     'drive/My Drive/Cui_workspace/Data/TweetParaphrase/commTweets/commTweets.NNP.tokenized.sampled.original.copynet',
                     'drive/My Drive/Cui_workspace/Data/TweetParaphrase/commTweets/commTweets.sampled.copynet.report', repeatTimes=1, split=False)

    verifyKeyComponents('drive/My Drive/Cui_workspace/Data/TweetParaphrase/commTweets/commTweets.NNP.tokenized.sampled.original.copynet',
                        'drive/My Drive/Cui_workspace/Data/TweetParaphrase/commTweets/commTweets.NNP.tokenized.sampled.items',
                        'drive/My Drive/Cui_workspace/Data/TweetParaphrase/commTweets/commTweets.sampled.original.contained.copynet.results', repeatTimes=1)
