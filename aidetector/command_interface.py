"""

In here, we have the functionality required for things in general as well as calling this from the command interface.


Imports:
    aidetector.aidetectorclass: Contains the AiDetector class used for creating the detection model.
    aidetector.inference: Contains functions for performing inference on trained models.
    aidetector.training: Contains functions for training models.
    aidetector.tokenization: Contains functions for tokenizing input data.
    argparse: Standard library for parsing command-line arguments.
    Halo: Library used for creating terminal spinners.

"""

from aidetector.aidetectorclass import *
from aidetector.inference import *
from aidetector.training import *
from aidetector.tokenization import *
import argparse
from halo import Halo

def main():

    """
    The main function of the module. Parses command-line arguments to determine if the module is in training or inference mode. 
    Depending on the mode, it calls the respective functions to perform training or inference.
    """
    parser = argparse.ArgumentParser(description='Training module for Generative AI text detection.')
    subparsers = parser.add_subparsers(dest='mode')

    # Training mode
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--datafile', type=str, required=True,help='File path to the data to train on.')
    parser_train.add_argument('--modeloutputfile', type=str, required=True,help='File path to save the trained model.')
    parser_train.add_argument('--vocaboutputfile', type=str, required=True,help='File path to save the vocabulary.')
    parser_train.add_argument('--tokenmodel',type=str,required=False,default="xx_ent_wiki_sm",help="The Spacy model to use for training.")
    parser_train.add_argument('--percentsplit',type=float,required=False,default=0.2,help="The percent split to use for training.")
    parser_train.add_argument('--classificationlabel',type=str,required=False,default="classification",help="The label in your dataset for classification. Default 'classification'")
    parser_train.add_argument('--textlabel',type=str,required=False,default="text",help="The label on your dataset for the text. Default 'text'")
    parser_train.add_argument('--download', dest='download', action='store_true', help='Download the model you have specified or the default of xx_ent_wiki_sm')
    parser_train.add_argument('--lowerbound',type=float,required=False,default=0.4,help="The lower bound for your Training Validation Loss (defaoult 0.4).")
    parser_train.add_argument('--upperbound',type=float,required=False,default=0.6,help="The upper bound for your Training Validation Loss (default 0.6)")
    parser_train.add_argument('--epochs',type=int,required=False,default=100,help="The maximum number of epochs for training the model (default 100)")


    parser_train.set_defaults(download=False) 

    # Inference mode
    parser_infer = subparsers.add_parser('infer')
    parser_infer.add_argument('--modelfile', type=str, required=True,help='File path of the trained model.')
    parser_infer.add_argument('--vocabfile', type=str, required=True,help='File path of the vocabulary.')
    parser_infer.add_argument('--text', type=str, required=True,help='Text string to classify.')
    parser_infer.add_argument('--tokenmodel',type=str,required=False,default="xx_ent_wiki_sm",help="The Spacy model to use for training. defaults to xx_ent_wiki_sm")
    parser_infer.add_argument('--threshold',type=float,required=False,default=0.5,help="The threshold to determine if the input is AI or not. Default 0.5")
    parser_infer.add_argument('--download', dest='download', action='store_true', help='Download the model you have specified or the default of xx_ent_wiki_sm')
    
    parser_infer.set_defaults(download=False) 
    args = parser.parse_args()

    if args.mode == 'train':
        tokenizer=get_tokenizer(args.tokenmodel,args.download)
        traintxt, test_text, train_labels, test_labels = load_data(args.datafile,percentsplit=args.percentsplit,classificationlabel=args.classificationlabel,textlabel=args.textlabel)
        vocab, trainseqs, testseqs = tokenize_data(
            traintxt,
            test_text,
            tokenizer
            
        )
        model = AiDetector(len(vocab))
        # Pass a sample input through the model to compute the size of the convolutional layer output
        _ = model(torch.zeros(1, trainseqs.size(1)).long())
        model.add_fc_layer()
        spinner = Halo(text='Training model', spinner='dots')
        spinner.start()

        train_model(model,
                    trainseqs,
                    torch.tensor(train_labels.values, dtype=torch.float),
                    testseqs,
                    torch.tensor(test_labels.values, dtype=torch.float),
                    epochs=args.epochs,
                    lowerbound=args.lowerbound,
                    upperbound=args.upperbound
        )
        spinner.stop()

        #print("training complete. Saving...")
        
        save_model(
            model, args.modeloutputfile
        )

        # Save the vocabulary
        save_vocab(vocab=vocab, vocaboutputfile=args.vocaboutputfile)


    elif args.mode == 'infer':
        #load the tokenizer
        tokenizer=get_tokenizer(args.tokenmodel,download=args.download)
        # Load the vocabulary
        vocab=load_vocab(args.vocabfile)

        model = AiDetector(len(vocab))
        
        model.load_state_dict(torch.load(args.modelfile))
        isai=check_input(
            model,
            vocab,
            args.text,
            tokenizer=tokenizer,
            threshold=args.threshold,
        )
        if isai:
            print("This was written by AI")
        else:
            print("This was written by a human.")
    else:
        parser.print_help()

if __name__ == "__main__":
    """
    Main entry point of the module. Calls the main function to start the training or inference process.
    """
    main()