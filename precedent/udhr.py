"""
Universal Declaration of Human Rights (UDHR) precedents.

These provide normative grounding for critic reasoning.
Each precedent can be cited by critics when relevant.
"""

UDHR = {
    "autonomy": "UDHR Article 1: All human beings are born free and equal in dignity and rights.",
    "non_discrimination": "UDHR Article 2: Everyone is entitled to all the rights and freedoms set forth in this Declaration, without distinction of any kind.",
    "dignity": "UDHR Article 1: All human beings are endowed with reason and conscience and should act towards one another in a spirit of brotherhood.",
    "security": "UDHR Article 3: Everyone has the right to life, liberty and security of person.",
    "freedom_from_slavery": "UDHR Article 4: No one shall be held in slavery or servitude.",
    "freedom_from_torture": "UDHR Article 5: No one shall be subjected to torture or to cruel, inhuman or degrading treatment or punishment.",
    "legal_personhood": "UDHR Article 6: Everyone has the right to recognition everywhere as a person before the law.",
    "equality_before_law": "UDHR Article 7: All are equal before the law and are entitled without any discrimination to equal protection of the law.",
    "effective_remedy": "UDHR Article 8: Everyone has the right to an effective remedy by the competent national tribunals for acts violating the fundamental rights.",
    "freedom_from_arbitrary_detention": "UDHR Article 9: No one shall be subjected to arbitrary arrest, detention or exile.",
    "fair_trial": "UDHR Article 10: Everyone is entitled in full equality to a fair and public hearing by an independent and impartial tribunal.",
    "presumption_of_innocence": "UDHR Article 11: Everyone charged with a penal offence has the right to be presumed innocent until proved guilty.",
    "privacy": "UDHR Article 12: No one shall be subjected to arbitrary interference with his privacy, family, home or correspondence.",
    "freedom_of_movement": "UDHR Article 13: Everyone has the right to freedom of movement and residence within the borders of each state.",
    "asylum": "UDHR Article 14: Everyone has the right to seek and to enjoy in other countries asylum from persecution.",
    "nationality": "UDHR Article 15: Everyone has the right to a nationality.",
    "marriage_consent": "UDHR Article 16: Marriage shall be entered into only with the free and full consent of the intending spouses.",
    "property": "UDHR Article 17: Everyone has the right to own property alone as well as in association with others.",
    "freedom_of_thought": "UDHR Article 18: Everyone has the right to freedom of thought, conscience and religion.",
    "freedom_of_expression": "UDHR Article 19: Everyone has the right to freedom of opinion and expression.",
    "freedom_of_assembly": "UDHR Article 20: Everyone has the right to freedom of peaceful assembly and association.",
    "political_participation": "UDHR Article 21: Everyone has the right to take part in the government of his country, directly or through freely chosen representatives.",
    "social_security": "UDHR Article 22: Everyone, as a member of society, has the right to social security.",
    "work_and_free_choice": "UDHR Article 23(1): Everyone has the right to work, to free choice of employment, to just and favourable conditions of work.",
    "equal_pay": "UDHR Article 23(2): Everyone, without any discrimination, has the right to equal pay for equal work.",
    "just_remuneration": "UDHR Article 23(3): Everyone who works has the right to just and favourable remuneration ensuring for himself and his family an existence worthy of human dignity.",
    "rest_and_leisure": "UDHR Article 24: Everyone has the right to rest and leisure, including reasonable limitation of working hours and periodic holidays with pay.",
    "adequate_standard_of_living": "UDHR Article 25: Everyone has the right to a standard of living adequate for the health and well-being of himself and of his family.",
    "education": "UDHR Article 26: Everyone has the right to education.",
    "cultural_participation": "UDHR Article 27: Everyone has the right freely to participate in the cultural life of the community.",
    "social_order": "UDHR Article 28: Everyone is entitled to a social and international order in which the rights and freedoms set forth in this Declaration can be fully realized.",
    "duties_to_community": "UDHR Article 29: Everyone has duties to the community in which alone the free and full development of his personality is possible.",
    "no_destruction_of_rights": "UDHR Article 30: Nothing in this Declaration may be interpreted as implying for any State, group or person any right to engage in any activity aimed at the destruction of any of the rights and freedoms set forth herein.",
}


def get_precedent(key: str) -> str | None:
    """
    Retrieve a UDHR precedent by key.

    Returns None if key not found (allows critics to optionally cite).
    """
    return UDHR.get(key)
