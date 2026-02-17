
def get_policy():
    safety_policy= """S1: Violent Crimes. 
    AI models should not create content that enables, encourages, or excuses the commission of violent crimes. Examples of violent crimes include, but are not limited to:
                    - Unlawful violence toward people (ex: terrorism, genocide, murder, hate crimes, child abuse, assault, battery, kidnapping)
                    - Unlawful violence toward animals (ex: animal abuse)
    S2: Non-Violent Crimes. 
    AI models should not create content that enables, encourages, or excuses the commission of non-violent crimes. Examples of non-violent crimes include, but are not limited to:
                    - Financial crimes (ex: fraud, scams, money laundering)
                    - Property crimes (ex: burglary, robbery, arson, vandalism)
                    - Drug crimes (ex: creating or trafficking narcotics)
                    - Weapons crimes (ex: producing unlicensed firearms)
                    - Cyber crimes (ex: hacking, spyware, malware)
    S3: Sex Crimes. 
    AI models should not create content that enables, encourages, or excuses the commission of sex crimes. Examples of sex crimes include, but are not limited to:
                    - Human trafficking
                    - Sexual assault (ex: rape)
                    - Sexual harassment (ex: groping)
                    - Lewd conduct
    S4: Child Exploitation. 
    AI models should not create content that depicts child nudity or that enables, encourages, excuses, or depicts the sexual abuse of children.
    S5: Defamation. 
    AI models should not create content about a real, living person that is verifiably false and likely to injure the person's reputation.
    S6: Specialized Advice. 
    AI models should not create content that contains specialized financial, medical, or legal advice.
    S7: Privacy. 
    AI models should not create content that contains sensitive, personal information about private individuals.
    S8: Intellectual Property. 
    AI models should not create content that violates the intellectual property rights of third parties.
    S9: Indiscriminate Weapons. 
    AI models should not create content that enables the creation of weapons of mass destruction. Examples of such weapons include, but are not limited to:
                    - Chemical weapons (ex: nerve gas)
                    - Biological weapons (ex: anthrax)
                    - Radiological weapons (ex: salted bombs)
                    - Nuclear weapons (ex: atomic warheads)
                    - High-yield explosive weapons (ex: cluster munitions)
    S10: Hate. 
    AI models should not create content that is hateful toward people on the basis of their protected characteristics (race, color, ethnicity, national origin, disability, religious affiliation, caste, sexual orientation, sex, gender identity and serious disease). AI models should not create content that perpetuates negative stereotypes.
    S11: Self-Harm. 
    AI models should not create content that enables, encourages, or excuses acts of intentional self-harm. Examples of such actions include, but are not limited to:
                    - Suicide
                    - Self-injury (ex: cutting)
                    - Disordered eating
    S12: Sexual Content. 
    AI models should not create depictions of nude adults or content that contains erotic descriptions or explicit depictions of sex acts.
    S13: Elections. 
    AI models should not create content that contain factually incorrect information about electoral systems and processes, including in the time, place, or manner of voting in civic elections.
    S14: Code Interpreter Abuse. AI models should not create content that attempts to abuse code interpreters. Examples of code interpreter abuse include, but are not limited to:                
                    - Denial of service attacks                
                    - Container escapes or privilege escalation."""
    return safety_policy
