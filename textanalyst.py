"""
filename: text_analyst.py

description: A reusable Python library for text analysis and comparison

author: Ananda Francis
"""

# import necessary libraries
from pathlib import Path
from collections import defaultdict
from collections import Counter
import re
import pandas as pd
import sankey as sk
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests


#%%
class TextAnalyst: 

    
    def __init__(self):
        """Constructor"""
        
        # intialize state dictionary used to store all registered text and their stats
        self.data = defaultdict(dict)
        
   
    
    def _avg_length(text_data, text_type):
        '''
        Purpose:
            Find average word/sentence/chapter length of a file
            
        Args:
            text_data (list): list of words, sentences or chapters cleaned out from text file
            text_type (str): define what text type composes the list text_data 
                            (chapters, sentences, or words)
            
        Returns:
             (int): Avg word/sentence/chapter length
        '''
        
        # intialize 
        total = []
        
        # turn each sentence into list of words (creating a list of lists)
        if text_type == 'sents':
            text_data = [sent.split() for sent in text_data]
         
        # leave list of words as is
        elif text_type == 'words':
            text_data = text_data
         
        # turn each chapter to a list of sentences (creating a list of lists)
        elif text_type == 'chaps':
            punc = '[.?!]'
            clean_text_data = []
            
            for chap in text_data:
                clean_chap = []
                chap = re.split(punc, chap)
                for sent in chap:
                    sent = sent.strip()
                    if sent != '':
                        clean_chap.append(sent)
                
                clean_text_data.append(clean_chap)
            
            text_data = clean_text_data
            
            
        # add length of each word/sentence/chapter to list  
        for words in text_data:
            total.append(len(words))
                    
        # find avg of all lengths in list 
        return sum(total) / len(total)
        
        
    
    def _get_chapters(file, title, parser_type): 
        '''
        Purpose:
            Clean up .txt files into list of chapters
            
        Args:
            file (str): filename being cleaned to register and load to dictionary 
                        or file itself as str from url
            title (str): title of Bible chapter
            
        Returns:
            chapters (list): list of chapters (each chapter is a str)
        '''
        
        if parser_type == 'bible_txt':
            # Read text file as a string
            text = Path(file).read_text()
            
        elif parser_type == 'bible_url':
            text = file
        
        # remove Book title from text
        text = re.sub(title, '', text)
        
        # intialize
        chapters = []
        
        # remove /n character
        text = re.sub('\n', ' ', text).strip().lower()
        
        # remove verse #s
        text = re.sub('\d+: ', ' ', text)
        
        # intialize (will be used laster)
        last_chap_num = 0
        
        
        for i in range(1,1000):
            
            # find where the next chapter starts (that will be where current chapter ends)
            match = re.search(str(i+1)+': \w*', text)
            if type(match) == re.Match: 
                stop_word = match.group()
                
                # group current chapter as a str and remove beginning of next chapter 
                chapter = re.search(str(i)+': .* '+stop_word, text).group()
                chapter = re.sub(stop_word, '', chapter)
                
                # clean text and add to list of chapters
                chapter = re.sub('\d+: ', '', chapter).strip().lower()
                chapters.append(chapter)
            
            # break loop once you reach the last chapter (because there is no next chapter for stop word)
            else: 
                last_chap_num = i
                
                break
            
        # get last chapter
        last_word = text.split()[-1]
        last_chap = re.search(str(last_chap_num)+': .*'+last_word, text).group()
        last_chap = re.sub('\d+: ', '', last_chap).strip().lower()
        chapters.append(last_chap)
        
            
        return chapters
    
    
    def _get_sentences(file, title, parser_type):
        '''
        Purpose:
            Clean up .txt files into lists of sentences
            
        Args:
            file (str): filename being cleaned to register and load to dictionary 
                        or file itself as str from url
            title (str): title of Bible chapter
            
        Returns:
            sents (list): list of sentences (each sentence is a str)
        '''
        
        # Read .txt file as a string
        if parser_type == 'bible_txt':
            text = Path(file).read_text()
            
            # remove Book title from text
            text = re.sub(title, '', text)
            
        elif parser_type == 'bible_url':
            text = file
            
            # remove Book title from text
            text = re.sub(title, '', text)
            
        elif parser_type == 'default':
            
            text = Path(file).read_text()
        
        # intialize 
        sents = []
            
        
        
            
        # create list of setences
        punc = '[.?!]'
        sentences = re.split(punc, text)
            
            
        # clean text for each sentence
        for sent in sentences: 
            sent = sent.strip().lower()
            sent = re.sub('\n', ' ', sent)
            sent = re.sub('\d+:\d+: ', '', sent)
                
            # only add sentenes, not blank strings to list of sentences
            blank = ''
            if sent != blank: 
                sents.append(sent)  
                
                
        return sents
                    
    
    
    @staticmethod
    def _default_parser(filename):
        '''
        Purpose:
            Clean up any .txt files removing unnecessary whitespace, punctuation, and capitalization
            and add statistics, words, chapters and sentences to dictionary 
            
        Args:
            filename (str): file being cleaned to register and load to dictionary
            title (str): title of Bible chapter
            
        Returns:
            results (dict): important statistics and storing of 
                            cleaned data in different formats
        '''
       
        # create list of sentences
        sents = TextAnalyst._get_sentences(filename, None, 'default')
        
        # create list of words
        words = [word for sent in sents for word in sent.split()]
        
        results = {
            
            'wordcount': Counter(words),
            
            'all_words': words,
            'all_sents': sents,
            
            'avg_sent_length': round(TextAnalyst._avg_length(sents, 'sents'), 2),
            'avg_word_length': round(TextAnalyst._avg_length(words, 'words'), 2),
            
            'no_of_unique_words': len(set(words)),
            'no_of_sentences': len(sents)
        }
        
        return results
        
        
    
    @staticmethod
    def _bible_txt_parser(filename, title):
        '''
        Purpose:
            Clean up Bible .txt files removing unnecessary whitespace, punctuation, and capitalization
            and add statistics, words, chapters and sentences to dictionary 
            
        Args:
            filename (str): file being cleaned to register and load to dictionary
            title (str): title of Bible chapter
            
        Returns:
            results (dict): important statistics and storing of 
                            cleaned data in different formats
        '''
        
        # create list of chapters
        chapters = TextAnalyst._get_chapters(filename, title, 'bible_txt')
        
        # create list of sentences
        sents = TextAnalyst._get_sentences(filename, title, 'bible_txt')

        # create list of words
        words = [re.sub('\W', '', word) for sent in sents for word in sent.split()]
        
        # create dict of statistics
        results = {
            'wordcount': Counter(words),
            
            'all_words': words,
            'all_sents': sents,
            'all_chaps': chapters,
            
            'avg_sent_length': round(TextAnalyst._avg_length(sents, 'sents'), 2),
            'avg_word_length': round(TextAnalyst._avg_length(words, 'words'), 2),
            'avg_chap_length': round(TextAnalyst._avg_length(chapters, 'chaps'), 2),
            
            'no_of_unique_words': len(set(words)),
            'no_of_sentences': len(sents),
            'no_of_chaps': len(chapters)
        }
        
        return results
        

    
    @staticmethod
    def _bible_url_parser(file_link, title): 
        '''
        Purpose:
            Extract text from url and clean up text to add statistics, words, 
            chapters and sentences to dictionary. Carry out the parsing and pre-processing 
            of your unique files (non .txt files)
            
        Args:
            filename (str): file being cleaned to register and load to dictionary
            title (str): title of Bible chapter
            
        Returns:
            results (dict): important statistics and storing of 
                            cleaned data in different formats
        '''
        
        # turn html link into str of text
        file = requests.get(file_link)
        
        # remove any text before title
        text = re.sub('.*[\n]*'+title, title, file.text)
        
        # create list of chapters
        chapters = TextAnalyst._get_chapters(text, title, 'bible_url')
        
        # create list of sentences
        sents = TextAnalyst._get_sentences(text, title, 'bible_url')

        # create list of words
        words = [re.sub('\W', '', word) for sent in sents for word in sent.split()]
        
        # create dict of statistics
        results = {
            'wordcount': Counter(words),
            
            'all_words': words,
            'all_sents': sents,
            'all_chaps': chapters,
            
            'avg_sent_length': round(TextAnalyst._avg_length(sents, 'sents'), 2),
            'avg_word_length': round(TextAnalyst._avg_length(words, 'words'), 2),
            'avg_chap_length': round(TextAnalyst._avg_length(chapters, 'chaps'), 2),
            
            'no_of_unique_words': len(set(words)),
            'no_of_sentences': len(sents),
            'no_of_chaps': len(chapters)
        }
        
        return results
        
        
    
    def _save_results(self, label, results):
        '''
        Purpose:
            Add statistics dictionary of each registered text file to state dictionary
            
        Args:
            label (str): label used to identity file in visualizations and dictionary
            results (dict): dictionary with stats and cleaned text formats of specific file
            
        Returns:
            None, adds results (dict) to self.data dictionary
        '''
        
        for k, v in results.items():
            self.data[k][label] = v
    
    
    
    def load_text(self, file, title=None, label=None, parser=None): 
        ''' 
        Purpose:
            Load or register text file within library. Should be able
            to store an arbitrary number of text files 
        
        Args:
            file (str): name of file user is registering or link to file
            title (str): title of Bible chapter
            label (str): label used to identity file in visualizations and dictionary
            parser (method): func/method used to parse data
            
        Return:
            None, adds registered text file to state variable self.data (dict)
        
        '''
        
        # intialize
        results = 0
        
        # normal parsing technique (for all .txt files)
        if parser is None:
            results = TextAnalyst._default_parser(file)
            
            # add registered label to state dictionary object
            self._save_results(label, results)
        
        # unique parsing technique
        if parser == 'bible_txt':
            results = TextAnalyst._bible_txt_parser(file, title)
            
            # add registered label to state dictionary object
            self._save_results(label, results)
        
        # unique parsing technique
        elif parser == 'bible_url':
            results = TextAnalyst._bible_url_parser(file, title)
            
            # add registered label to state dictionary object
            self._save_results(label, results)
    
        # create label for visualizations and dictionary
        if label is None:
            label = file
    
    
    
    
    
    def generate_common_words(self, k): 
        '''
        Purpose:
            Create list of common words
            
        Args:
            k (int): number of top most common words from each file
            
        Returns:
            com_words (list): List of k most common words from each file, not including filler_words
        '''
        
        # intialize
        com_words = []
        
        # list of each registered text file
        text_labels = list(self.data['all_words'].keys())
        
        # create list of filler words
        filler_words = ['and','the','of','his','he','to', 'in', 'unto', 'i', 'that', 'of', 'my', 
                        'him', 'was', 'a', 'which', 'said', 'me', 'it', 'with', 'for', 'is', 'shall',
                        'as', 'be', 'not', 'thee', 'them', 'they', 'thou', 'thy', 'were', 'all', 'are', 
                        'from', 'have', 'o', 'their', 'will','ye', 'she', 'her', 'up', 'hath', 'let', 
                        'hast', 'but', 'out', 'upon', 'on', 'there', 'on', 'our', 'you', 'your', 'when',
                        'then', 'also', 'into', 'this', 'we', 'us']
        
        # create list of words not including filler words
        for text in text_labels:
            words = self.data['all_words'][text]
            non_filler_words = [word for word in words if word not in filler_words]
            
            # use Counter module to find most common words of each text file
            count = Counter(non_filler_words)
            topwords = [w for w,c in count.most_common(k)]
            
            # add k most common words to common word list
            for tword in topwords:
                com_words.append(tword)
            
        return com_words
        
    
    
    def wordcount_sankey(self, k=5): 
        '''
        Purpose:
            Create Sankey visualization from text name (or label) to common word where thickness 
            of line is the number of times that words occurs in the specified text
            
        Args:
            k (int): number of top most common words from each file
            
        Returns:
            None, generates Sankey visualization
        '''
    
        # intialize
        df = pd.DataFrame()
        
        
        # create common words list
        word_list = TextAnalyst.generate_common_words(self, k)
        
        # find # of times common word appears in each text
        wordcounts = self.data['wordcount']
        texts = wordcounts.keys()
        
        for text in texts:
            for word in word_list:
                c_word_count = wordcounts[text][word]
                series = pd.Series({'text': text, 'c_word': word})
                

                # add row of text and common word the # of times the common word appears in text
                for i in range(c_word_count):
                    df = df.append(series, ignore_index=True)
        
        # create Sankey Visualization
        sk.make_sankey(df, 'text', 'c_word', 'count', 0)
                
    
    
    def score_chaps(self, label, minpol=-1.0, maxpol=1.0):
        '''
        Purpose:
            Create a dictionary with keys representing each chapter
            and values representing polarity of that chapter
            
        Args:
            label (str): name used to identity specific registered text file
            minpol (float): minimum polarity value
            maxpol (float): maximum polarity value
            
        Returns:
            filtered (dict): Bible chapters and their polarity scores
        '''
        
        # intialize
        filtered = {}
        
        
        # create list of chapters
        chapters = self.data['all_chaps'][label]
        
        
        # update dictionary with keys (chapter) and values (polarity)
        for chap in chapters:
            pol, sub = TextBlob(chap).sentiment
            if minpol <= pol <= maxpol:
                filtered[chap] = pol
        
        # return dictionary
        return filtered
        
    
    
    def polarity_over_time(self):
        '''
        Purpose:
            Plot polarity over time (use subplots for each text file in this visualization)
            
        Args:
            None
            
        Returns:
            None, renders line subplots for each registered text file
        '''

        # all the registered texts
        texts = self.data['all_chaps'].keys()
        
        # create subplot # 
        for idx, text in enumerate(texts):
            
            # create x values
            scored_chaps = TextAnalyst.score_chaps(self, text)
            polarity = scored_chaps.values()
            
            # create y values
            chaps = scored_chaps.keys()
            time = []
            for index, value in enumerate(chaps):
                time.append(index)
            
    
            # plot lineplots
            n_rows = round((len(texts)/2))
            
            # factor in if you are plotting only one text
            if n_rows == 0:
                n_rows = 1
                
            plt.subplot(n_rows, 2, idx + 1)
            sns.lineplot(x=time, y=polarity, color='black')
            
            # label axises (only the outermost axises)
            if idx % 2 == 0:
                plt.ylabel('Polarity', fontsize=10)
                
            if idx >= (n_rows*2-2):
                plt.xlabel('Chapter Order', fontsize=10)
            
            # create title and consistent y axis range
            plt.title(text+': polarity over time (by chapter)', fontsize=16)
            plt.ylim(-1, 1)
            
        # groom entire figure
        plt.gcf().set_size_inches((15, 18))
        plt.show()


    
    def compare_avg_lengths(self):
        '''
        Purpose:
            Create bar graphs to compare average length of chapters, sentences and words
            of each Biblical book registered as a text file. Overlay 3 bar graph 
            visualizations from each text file and distinguish using labels across the x axis
            
        Args:
            None
        
        Returns:
            None, renders bar graph with each length comparison, overlayed together
        '''

        # create list of texts
        texts = list(self.data['avg_chap_length'].keys())
        
        # create dictionary for each bar plot
        avg_chap_length = list(self.data['avg_chap_length'].values())
        avg_sent_length = list(self.data['avg_sent_length'].values())
        avg_word_length = list(self.data['avg_word_length'].values())
        
        # create x axis arrangement
        x_axis = np.arange(len(texts))
        
        # plot each bar plot, overlayed on same graph
        plt.figure(figsize=(20, 10))
        
        plt.bar((x_axis-.3), avg_chap_length, .3, label='avg chap length (sents)', color='r')
        plt.bar(x_axis, avg_sent_length, .3, label='avg sent length (words)', color='b')
        plt.bar((x_axis+.3), avg_word_length, .3, label='avg word length (letters)', color=(.55, 0, .8))
        
        # customize visualization
        plt.xticks(x_axis, texts, fontsize=15)
        plt.title('Average Lengths for Chapters, Words and Sentences in Biblical Books', fontsize=40)
        plt.ylabel('Count', fontsize=30)
        plt.xlabel('Bible Book Name', fontsize=30)
        
        # label each bar with its height
        for i in range(len(texts)):
            plt.text(i-.35, avg_chap_length[i], str(avg_chap_length[i]), fontsize=12)
            plt.text(i-.1, avg_sent_length[i], str(avg_sent_length[i]), fontsize=12)
            plt.text(i+.2, avg_word_length[i], str(avg_word_length[i]), fontsize=12)
        
        
        plt.legend()
        plt.show()
        
        
    def compare_total_lengths(self):
            '''
            Purpose:
                Create bar graphs to compare total number of chapters, sentences and unique words
                of each Biblical book registered as a text file. Overlay 3 bar graph visualizations
                from each text file and distinguish using labels across the x axis
                
            Args:
                None
            
            Returns:
                None, renders bar graph with each length comparison, overlayed together
            '''
            

            # create list of texts
            texts = list(self.data['no_of_unique_words'].keys())
            
            # create dictionary for each bar plot
            no_of_chaps = list(self.data['no_of_chaps'].values())
            no_of_sentences = list(self.data['no_of_sentences'].values())
            no_of_unique_words = list(self.data['no_of_unique_words'].values())
            
            # create x axis arrangement
            x_axis = np.arange(len(texts))
            
            # plot each bar plot, overlayed on same graph
            plt.figure(figsize=(20, 10))
            
            plt.bar((x_axis-.3), no_of_chaps, .3, label='chapter count', color='r')
            plt.bar(x_axis, no_of_sentences, .3, label='sentence count', color='b')
            plt.bar((x_axis+.3), no_of_unique_words, .3, label='unique word count', color=(.55, 0, .8))
            
            # customize visualization
            plt.xticks(x_axis, texts, fontsize=15)
            plt.title('Number of Chapters, Unique Words and Sentences in Biblical Books', fontsize=40)
            plt.ylabel('Count', fontsize=30)
            plt.xlabel('Bible Book Name', fontsize=30)
            
            # label each bar with its height
            for i in range(len(texts)):
                plt.text(i-.35, no_of_chaps[i], str(no_of_chaps[i]), fontsize=12)
                plt.text(i-.1, no_of_sentences[i], str(no_of_sentences[i]), fontsize=12)
                plt.text(i+.2, no_of_unique_words[i], str(no_of_unique_words[i]), fontsize=12)
            
            
            plt.legend()
            plt.show()
        
