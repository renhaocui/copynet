
def createMirroLines(inputFilename, outputFilename):
    outFile = open(outputFilename, 'w')
    with open(inputFilename, 'r') as fr:
        for line in fr:
            data = line.strip()
            outFile.write(data + ' >>><<< ' + data + '\n')
    outFile.close()


if __name__ == '__main__':
    createMirroLines('data/commTweets.NNP.tokenized.sampled.original', 'data/commTweets.NNP.tokenized.sampled.original.data')