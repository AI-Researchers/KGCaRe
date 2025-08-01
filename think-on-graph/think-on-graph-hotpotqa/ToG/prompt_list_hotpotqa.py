extract_relation_prompt_wiki = """Please retrieve %s relations (separated by semicolon) that contribute to the question and rate their contribution on a scale from 0 to 1 (the sum of the scores of %s relations is 1).
Q: Mesih Pasha's uncle became emperor in what year?
Topic Entity: Mesih Pasha
Relations: ['child', 'country of citizenship', 'date of birth', 'family', 'father', 'languages spoken', 'written or signed', 'military rank', 'occupation', 'place of death', 'position held', 'religion or worldview', 'sex or gender', 'sibling', 'significant event']
A: 1. {family (Score: 0.5)}: This relation is highly relevant as it can provide information about the family background of Mesih Pasha, including his uncle who became emperor.
2. {father (Score: 0.4)}: Uncle is father's brother, so father might provide some information as well.
3. {position held (Score: 0.1)}: This relation is moderately relevant as it can provide information about any significant positions held by Mesih Pasha or his uncle that could be related to becoming an emperor.

Q: Van Andel Institute was founded in part by what American businessman, who was best known as co-founder of the Amway Corporation?
Topic Entity: Van Andel Institute
Relations: ['affiliation', 'country', 'donations', 'educated at', 'employer', 'headquarters location', 'legal form', 'located in the administrative territorial entity', 'total revenue']
A: 1. {affiliation (Score: 0.4)}: This relation is relevant because it can provide information about the individuals or organizations associated with the Van Andel Institute, including the American businessman who co-founded the Amway Corporation.
2. {donations (Score: 0.3)}: This relation is relevant because it can provide information about the financial contributions made to the Van Andel Institute, which may include donations from the American businessman in question.
3. {educated_at (Score: 0.3)}: This relation is relevant because it can provide information about the educational background of the American businessman, which may have influenced his involvement in founding the Van Andel Institute.

Q: """

score_entity_candidates_prompt_wiki = """Please score the entities' contribution to the question on a scale from 0 to 1 (the sum of the scores of all entities is 1).
Q: Staten Island Summer, starred what actress who was a cast member of "Saturday Night Live"?
Relation: cast member
Entites: Ashley Greene; Bobby Moynihan; Camille Saviola; Cecily Strong; Colin Jost; Fred Armisen; Gina Gershon; Graham Phillips; Hassan Johnson; Jackson Nicoll; Jim Gaffigan; John DeLuca; Kate Walsh; Mary Birdsong
Score: 0.0, 0.0, 0.0, 0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0
To score the entities\' contribution to the question, we need to determine which entities are relevant to the question and have a higher likelihood of being the correct answer.
In this case, we are looking for an actress who was a cast member of "Saturday Night Live" and starred in the movie "Staten Island Summer." Based on this information, we can eliminate entities that are not actresses or were not cast members of "Saturday Night Live."
The relevant entities that meet these criteria are:\n- Ashley Greene\n- Cecily Strong\n- Fred Armisen\n- Gina Gershon\n- Kate Walsh\n\nTo distribute the scores, we can assign a higher score to entities that are more likely to be the correct answer. In this case, the most likely answer would be an actress who was a cast member of "Saturday Night Live" around the time the movie was released.
Based on this reasoning, the scores could be assigned as follows:\n- Ashley Greene: 0\n- Cecily Strong: 0.4\n- Fred Armisen: 0.2\n- Gina Gershon: 0\n- Kate Walsh: 0.4

Q: {}
Relation: {}
Entites: """

prompt_evaluate_wiki="""Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer whether it's sufficient for you to answer the question with these triplets and your knowledge (Yes or No).
Q: Viscount Yamaji Motoharu was a general in the early Imperial Japanese Army which belonged to which Empire?
Knowledge Triplets: Imperial Japanese Army, allegiance, Emperor of Japan
Yamaji Motoharu, allegiance, Emperor of Japan
Yamaji Motoharu, military rank, general
A: {Yes}. Based on the given knowledge triplets and my knowledge, Viscount Yamaji Motoharu, who was a general in the early Imperial Japanese Army, belonged to the Empire of Japan. Therefore, the answer to the question is {Empire of Japan}.

Q: Who is the coach of the team owned by Steve Bisciotti?
Knowledge Triplets: psilocybin, described by source, Opium Law,
psilocybin, found in taxon, Gymnopilus purpuratus,
psilocybin, found in taxon, Gymnopilus spectabilis, 
Opium Law, part of, norcodeine (stereochemistry defined), 
Gymnopilus purpuratus, edibility, psychoactive mushroom,
Gymnopilus spectabilis, parent taxon, Gymnopilus
A: {No}. Based on the given knowledge triplets and my knowledge, the specific psychedelic compound found in the Psilocybin genus mushroom that is converted to psilocin by the body is not explicitly mentioned. Therefore, additional knowledge about the specific compounds and their conversion to psilocin is required to answer the question.

Q: Which tennis player is younger, John Newcombe or Květa Peschke?
Knowledge Triplets: Květa Peschke, date of birth, +1975-07-09T00:00:00Z, 
John Newcombe, date of birth, +1944-05-23T00:00:00Z,
John Newcombe, country of citizenship, Australia
A: {Yes}. Based on the given knowledge triplets and my knowledge, John Newcombe was born on May 23, 1944, and Květa Peschke was born on July 9, 1975. Therefore, {Květa Peschke} is younger than John Newcombe.

Q: At what stadium did Mychal George Thompson play home games with the San Antonio Spurs?
Knowledge Triplets: San Antonio Spurs, home venue, AT&T Center
San Antonio Spurs, home venue, Alamodome
San Antonio Spurs, home venue, Fort Worth Convention Center
AT&T Center, occupant, San Antonio Spurs
Fort Worth Convention Center, located in the administrative territorial entity, Texas
Fort Worth Convention Center, occupant, San Antonio Spurs
A: {Yes}. Based on the given knowledge triplets and my knowledge, Mychal George Thompson played home games with the San Antonio Spurs at the AT&T Center. Therefore, the answer to the question is {AT&T Center}.

"""

answer_prompt_wiki = """Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer the question with these triplets and your own knowledge.

Example:
Q: Viscount Yamaji Motoharu was a general in the early Imperial Japanese Army which belonged to which Empire?
Knowledge Triplets: Imperial Japanese Army, allegiance, Emperor of Japan
Yamaji Motoharu, allegiance, Emperor of Japan
Yamaji Motoharu, military rank, general
A: Empire of Japan.

Q: At what stadium did Mychal George Thompson play home games with the San Antonio Spurs?
Knowledge Triplets: San Antonio Spurs, home venue, AT&T Center
San Antonio Spurs, home venue, Alamodome
San Antonio Spurs, home venue, Fort Worth Convention Center
AT&T Center, occupant, San Antonio Spurs
Fort Worth Convention Center, located in the administrative territorial entity, Texas
Fort Worth Convention Center, occupant, San Antonio Spurs
A: AT&T Center.

Q:
"""

generate_directly = """Q: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
A: Washington, D.C..

Q: Who lists Pramatha Chaudhuri as an influence and wrote Jana Gana Mana?
A: Bharoto Bhagyo Bidhata.

Q: Who was the artist nominated for an award for You Drive Me Crazy?
A: Jason Allen Alexander.

Q: What person born in Siegen influenced the work of Vincent Van Gogh?
A: Peter Paul Rubens.

Q: What is the country close to Russia where Mikheil Saakashvii holds a government position?
A: Georgia.

Q: What drug did the actor who portrayed the character Urethane Wheels Guy overdosed on?
A: Heroin."""

score_entity_candidates_prompt_wiki = """Please score the entities' contribution to the question on a scale from 0 to 1 (the sum of the scores of all entities is 1).
Q: Staten Island Summer, starred what actress who was a cast member of "Saturday Night Live"?
Relation: cast member
Entites: Ashley Greene; Bobby Moynihan; Camille Saviola; Cecily Strong; Colin Jost; Fred Armisen; Gina Gershon; Graham Phillips; Hassan Johnson; Jackson Nicoll; Jim Gaffigan; John DeLuca; Kate Walsh; Mary Birdsong
Score: 0.0, 0.0, 0.0, 0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0
To score the entities\' contribution to the question, we need to determine which entities are relevant to the question and have a higher likelihood of being the correct answer.
In this case, we are looking for an actress who was a cast member of "Saturday Night Live" and starred in the movie "Staten Island Summer." Based on this information, we can eliminate entities that are not actresses or were not cast members of "Saturday Night Live."
The relevant entities that meet these criteria are:\n- Ashley Greene\n- Cecily Strong\n- Fred Armisen\n- Gina Gershon\n- Kate Walsh\n\nTo distribute the scores, we can assign a higher score to entities that are more likely to be the correct answer. In this case, the most likely answer would be an actress who was a cast member of "Saturday Night Live" around the time the movie was released.
Based on this reasoning, the scores could be assigned as follows:\n- Ashley Greene: 0\n- Cecily Strong: 0.4\n- Fred Armisen: 0.2\n- Gina Gershon: 0\n- Kate Walsh: 0.4

Q: {}
Relation: {}
Entites: """