# -*- coding: utf-8 -*-
import math
from nltk.corpus import stopwords
from nltk import pos_tag
import re
import spacy
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = re.sub('<.*?>', '', text).strip()
    text = re.sub('(\s)+', r'\1', text)
    text = re.sub(r'[^\w\s]','', text)
    return text

def preprocessing(doc,lsw):
    doc = clean_text(doc)
    texts = nlp(doc)
    tokens = []
    for token in texts:
        tokens.append(token.text)
    pos_tags = pos_tag(tokens)
    tokenized_doc = [word for word, tag in pos_tags if tag.startswith("N") or tag.startswith("V")]
    for item in lsw:
        tokenized_doc = list(filter((item).__ne__, tokenized_doc))
    return tokenized_doc

class Extractor():
    def __init__(self):
        self.stopwords = stopwords.words('english')

    def run(self,document,num_keywords):
        tokens = preprocessing(document,self.stopwords)
        # Calculate the position weights of the filtered words using a combination of linear and logarithmic scales
        max_position = len(tokens)
        position_weights = [ i / max_position + (1 - math.log(i + 1) / math.log(max_position)) for i, word in enumerate(tokens)]

        tf = {}
        for i,token in enumerate(tokens):
            if token[0].isupper():
                weight = 1.8  # Assign a weight of 1.8 to uppercase tokens
            else:
                weight = 1  # Assign a weight of 1 to lowercase tokens
            tf[token] = tf.get(token, 0) + weight * position_weights[i]

        # Get top keywords
        sorter = sorted(tf.items(), key=lambda x:x[1], reverse=True)
        top_keywords = list(dict(sorter[:num_keywords]).keys())
        top_keywords = [s.replace("_"," ") for s in top_keywords]
        return top_keywords

if __name__=='__main__':
    extractor = Extractor()
    keywords = extractor.run("""Robert Blake, actor known for "Baretta" and "Lost Highway," dies at 89. Actor Robert Blake, whose decades-long film and television career was tarnished by a notorious murder trial, has died at the age of 89. Blake died in Los Angeles, his niece Noreen confirmed to CBS News Thursday. She said he died after a battle with heart disease, adding that he "passed away peacefully with family and friends." The Los Angeles County coroner's office told CBS News that it "did not have a report" about Blake's death. "Due to his age and reported medical history his death may not fall under our jurisdiction," a statement from the coroner's office read.  Prior to being tried and acquitted in his wife's shooting death, Blake was best known for the 1970s television series "Baretta," for which he won a best actor Emmy in 1975, and his last screen role, the 1997 film "Lost Highway." Actor Robert Blake leaves the Burbank County Courthouse after appearing in court for the wrongful-death lawsuit filed against him by the children of his slain wife, Bonnie Lee Bakley, on Aug. 24, 2005, in Burbank, California. / GETTY IMAGES However, on May 4, 2001, Blake's wife, Bonny Lee Bakley, was shot and killed in Blake's car near a restaurant in the Studio City neighborhood of Los Angeles. Blake was arrested on a murder charge in April 2002. The case finally went to trial in late 2004, and Blake was acquitted by an L.A. jury in early 2005.  The jury of seven men and five women delivered the verdicts on its ninth day of deliberations, following a four-month trial with a cast of characters that included two Hollywood stuntmen who said Blake tried to hire them to kill his wife. However, no eyewitnesses, blood or DNA evidence linked Blake to the crime. The murder weapon, found in a trash bin, could not be traced to Blake, and witnesses said the minuscule amounts of gunshot residue found on Blake's hands could have come from a different gun he said he carried for protection. Blake had hundreds of film and television credits. His career began when he was a preschooler, with the role of Mickey in the 1930s and 1940s kids' comedy film series "Our Gang," which was re-run for decades on television. He won critical acclaim for his portrayal of real-life killer Perry Smith in the 1967 film "In Cold Blood." In 1993, Blake won another Emmy as the title character in, "Judgment Day: The John List Story," portraying a soft-spoken, churchgoing man who murdered his wife and three children — also based on the true story of a convicted murderer. He was born Michael James Gubitosi on Sept. 18, 1933, in Nutley, New Jersey. His father, an Italian immigrant and his mother, an Italian American, wanted their three children to succeed in show business. At age 2, Blake was performing with a brother and sister in a family vaudeville act called, "The Three Little Hillbillies." When his parents moved the family to L.A., his mother found work for the kids as movie extras, and little Mickey Gubitosi was plucked from the crowd by producers who cast him in the "Our Gang" comedies. He appeared in the series for five years and changed his name to Bobby Blake. He went on to work with Hollywood legends, playing the young John Garfield in "Humoresque" in 1946 and the little boy who sells Humphrey Bogart a crucial lottery ticket in "The Treasure of the Sierra Madre." In adulthood, he landed serious movie roles. The biggest breakthrough was in 1967 with "In Cold Blood." Later there were films including, "Tell Them Willie Boy is Here" and "Electra Glide in Blue." In 1961, Blake and actress Sondra Kerr married and had two children, Noah and Delinah. They divorced in 1983. His fateful meeting with Bakley came in 1999 at a jazz club where he went to escape loneliness. "Here I was, 67 or 68 years old. My life was on hold. My career was stalled out," he said in an interiew with The Associated Press. "I'd been alone for a long time." When Bakley gave birth to a baby girl, she named Christian Brando — son of Marlon — as the father. But DNA tests pointed to Blake. Blake first saw the little girl, named Rosie, when she was two months old and she became the focus of his life. He married Bakley because of the child. "Rosie is my blood. Rosie is calling to me," he said. "I have no doubt that Rosie and I are going to walk off into the sunset together." Prosecutors would claim that he planned to kill Bakley to get sole custody of the baby and tried to hire hitmen for the job. But evidence was muddled and a jury rejected that theory. On her last night alive, Blake and his 44-year-old wife dined at a neighborhood restaurant, Vitello's. He claimed she was shot when he left her in the car and returned to the restaurant to retrieve a handgun he had inadvertently left behind. Police were initially baffled and Blake was not arrested until a year after the crime occurred. Once a wealthy man, he spent millions on his defense and wound up living on social security and a Screen Actor's Guild pension. In a 2006 interview with the AP a year after his acquittal, Blake said he hoped to restart his career. "I'd like to give my best performance," he said. "I'd like to leave a legacy for Rosie about who I am. I'm not ready for a dog and fishing pole yet. I'd like to go to bed each night desperate to wake up each morning and create some magic." Trending News Oscars 2023: List of winners nominees Hugh Grant's Oscars interview with Ashley Graham goes viral Fellow "Goonies" stars celebrate Ke Huy Quan's Oscar win Rapper Costa Titch dies after collapsing on stage In: Obituary""",
    10)
    print(keywords)