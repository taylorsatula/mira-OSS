/*
 * repulsion-feedback.js
 *
 * Owns the "Repulsed!" feedback feature: a regex bank scans each completed
 * assistant response for known AI-tells (validation reflexes, closing offers,
 * sportscasting, etc.). When a tell matches, the backend captures it immediately
 * and a bright pink toolbar indicator reveals. Clicking the indicator opens a
 * popover where the user can add a human reason.
 *
 * Design principles:
 *   - Prefer multi-word phrases over single words (higher signal).
 *   - Use negative lookahead to exclude legitimate uses.
 *   - For structural tells (em-dash clusters), require thresholds.
 *   - Better to miss some tells than flood with false positives.
 *
 * Captured fields:
 *   - reason: the user's free-form text
 *   - response_text: the offending assistant response
 *   - preceding_user_message: the user's prompt that elicited the response
 *   - matched_tells: which tell-names fired
 */
(function () {
    'use strict';

    const TELLS = [
        // ── Validation Reflex (#1) & Praise Before Substance (#2) ──────────────
        // Model validates, praises, or emotionally endorses the user before answering.

        { name: 'great_question',
          pattern: /\bgreat question\b/i },

        { name: 'youre_absolutely_right',
          pattern: /\byou(?:'?re| are) absolutely right\b/i },

        { name: 'fascinating_thing',
          pattern: /\b(?:that'?s |what )?(?:a |an )?(?:fascinating|interesting|excellent|wonderful|brilliant|insightful|thoughtful) (?:question|point|observation|perspective|insight|take)\b/i },

        { name: 'sharp_observation',
          pattern: /\b(?:sharp|sophisticated|perceptive|keen) (?:observation|eye|instinct|intuition)\b/i },

        { name: 'love_this_direction',
          pattern: /\bI love (?:this direction|where this is going|this approach)\b/i },

        { name: 'good_instinct',
          pattern: /\b(?:good|great|excellent) instinct\b/i },

        { name: 'asking_right_question',
          pattern: /\byou'?re asking exactly the right question\b/i },

        { name: 'spot_on',
          pattern: /\bspot[- ]?on\b/i },

        // You make a good point / raise a valid point / bring up an interesting point
        { name: 'you_make_a_point',
          pattern: /\byou (?:make|raise|bring up) (?:a |an )(?:good|great|valid|excellent|fair|interesting) point\b/i },

        // ── Appreciation & Eager-to-Help Openings (#50, #88) ───────────────────
        // Synthetic gratitude and cheerful compliance wrappers.

        // I appreciate your patience / that / the perspective
        { name: 'i_appreciate',
          pattern: /\bI appreciate (?:your|that|the)\b/i },

        // I'd be happy/glad/delighted/more than happy to
        { name: 'id_be_happy',
          pattern: /\bI'?d be (?:happy|glad|delighted|more than happy) to\b/i },

        // Absolutely, here's... / Sure, here's... / Of course! / Happy to!
        { name: 'cheerful_compliance',
          pattern: /\b(?:absolutely|sure|of course|certainly)[,.]\s*(?:here'?s|here is)\b/i },

        // ── Closing Offer Ritual (#7) & Landing Compulsion (#129, #140) ────────
        // Unnecessary service-offer tails and closure sentences after the answer is done.

        { name: 'closer_hope_helps',
          pattern: /\b(?:I hope|hope) (?:this|that) helps\b/i },

        { name: 'closer_feel_free',
          pattern: /\bfeel free to (?:ask|reach out|let me know|contact)\b/i },

        { name: 'closer_let_me_know',
          pattern: /\blet me know if you (?:have any|need|want|'?d like)\b/i },

        { name: 'closer_dont_hesitate',
          pattern: /\bdon'?t hesitate to (?:ask|reach out|contact|let me know)\b/i },

        // That gives you a strong foundation / You're in a good place to move forward
        { name: 'landing_compulsion',
          pattern: /\b(?:that gives you|you'?re in) (?:a strong foundation|a good place|a solid starting point|the foundation)\b/i },

        // That's the core of it / Everything else follows from that / This is the right path forward
        { name: 'response_finality_theater',
          pattern: /\b(?:that'?s the core of it|everything else follows|this is the right path forward|that'?s everything you need)\b/i },

        // ── Didactic Staging (#5) & Instruction-Following Announcement (#90) ───
        // Model announces teaching mode or style instead of embodying it.

        // Let's break it down / Let us break it down / unpack this / Here's the key idea / Think of it this way
        { name: 'didactic_staging',
          pattern: /\b(?:let'?s|let us) (?:break (?:it|this) down|unpack this|walk through|go through|dig into)\b/i },

        // The key idea / The key takeaway / The main thing
        { name: 'key_idea_announcement',
          pattern: /\bthe key (?:idea|takeaway|thing to understand|thing to remember)\b/i },

        // Think of it this way / A good mental model / Think of it like
        { name: 'mental_model_framing',
          pattern: /\b(?:think of it (?:this way|like)|a good mental model|the mental model is)\b/i },

        // I'll be direct / I'll keep this concise / No fluff / I won't sugarcoat it
        { name: 'instruction_following_announcement',
          pattern: /\bI'?ll (?:be direct|keep this concise|get straight to the point|skip the fluff|not sugarcoat)\b/i },

        // ── Over-Structured Helpfulness (#6) & Fake Decomposition (#17) ────────
        // Forces tidy frameworks, numbered lists, or invented categories.

        // Key takeaways / Action plan / Pros and cons / Here are X ways
        { name: 'over_structured_helpfulness',
          pattern: /\b(?:key takeaways|action plan|pros and cons|things to consider|ways to think about it)\b/i },

        // There are three layers here / This breaks down into / The problem has two parts
        { name: 'fake_decomposition',
          pattern: /\b(?:there are \d+ (?:layers|parts|pieces|things going on)|this breaks down into|we can break this down)\b/i },

        // Short version: / Long version: / TL;DR: / The short answer is
        { name: 'short_long_version_ritual',
          pattern: /\b(?:short version|long version|TL;DR|the short answer|the long answer)[\s:]/i },

        // ── Reassurance Padding (#8) ───────────────────────────────────────────
        // Cushions ordinary technical exchanges with emotional reassurance.

        // You're not crazy / That makes sense / You're on the right track
        { name: 'reassurance_padding',
          pattern: /\byou'?re (?:not crazy|on the right track|thinking along the right lines|heading in the right direction)\b/i },

        // That makes sense / That's completely understandable / That's totally fair
        { name: 'makes_sense_reassurance',
          pattern: /\b(?:that makes sense|that'?s completely understandable|that'?s totally fair|your frustration makes sense)\b/i },

        // ── False Intimacy / Companion Voice (#9) ──────────────────────────────
        // Implies camaraderie, personal involvement, or shared emotional investment.

        // I'm excited to help / I love where this is going / We can tackle this together
        { name: 'false_intimacy',
          pattern: /\b(?:I'?m excited to help|I love where this is going|we can tackle this together|I'?m with you on this)\b/i },

        // ── Apology Reflex (#12) & Apology Substitution (#132) ─────────────────
        // Should simply correct and continue instead.

        // Sorry about that / Apologies for the confusion / Thanks for catching that
        { name: 'apology_reflex',
          pattern: /\b(?:sorry about that|apologies for the confusion|my apologies|I should have been clearer|I missed that)\b/i },

        // Thanks for catching that / Fair correction / You're right, my mistake
        { name: 'apology_substitution',
          pattern: /\b(?:thanks for catching that|fair correction|you'?re right, my mistake|good catch, you'?re right)\b/i },

        // ── Synthetic Gratitude (#105) ─────────────────────────────────────────
        // Thanks as social ritual rather than genuine appreciation.

        // Thanks for clarifying / sharing / the context / the correction
        { name: 'synthetic_gratitude',
          pattern: /\bthanks for (?:clarifying|sharing|the context|the correction|pointing that out|bringing this up)\b/i },

        // ── Fake Collaboration (#28) ───────────────────────────────────────────
        // Pretends to share agency or project ownership.

        // We should / Our goal / Let's build / We need to
        { name: 'fake_collaboration',
          pattern: /\b(?:we should|our goal|let'?s build|we need to|we'?ll need to)\b/i },

        // ── Explanatory Throat-Clearing (#66) ──────────────────────────────────
        // Begins with filler before the answer.

        // Basically / Essentially / Fundamentally / Ultimately / At its core
        { name: 'throat_clearing',
          pattern: /(?:^|\n\s*|[.!?]\s*)(?:basically|essentially|fundamentally|ultimately|at its core|in essence|put simply),?\s/i },

        // ── Artificial Escalation (#68) ────────────────────────────────────────
        // Dramatizes hierarchy or profundity.

        // The deeper issue / The real problem / At a deeper level / The bigger concern
        { name: 'artificial_escalation',
          pattern: /\b(?:the deeper issue|the real problem|the bigger concern|at a deeper level|the underlying issue|the root cause)\b/i },

        // ── Assistant Narrator Mode (#70) ──────────────────────────────────────
        // Narrates what it is about to do.

        // I'll walk through / I'll explain / I'll outline / I'll frame this as
        { name: 'assistant_narrator',
          pattern: /\bI'?ll (?:walk through|explain|outline|frame this|go over|break down)\b/i },

        // ── Unrequested Empathy in Technical Contexts (#72) ────────────────────
        // Mirrors emotion in technical work instead of addressing the issue.

        // That sounds frustrating / I can see why that's annoying / That's a pain
        { name: 'emotional_mirroring_technical',
          pattern: /\b(?:that sounds (?:frustrating|annoying|incredibly frustrating)|I can see why that'?s (?:annoying|frustrating|so frustrating)|that must be annoying|that'?s a pain)\b/i },

        // ── Corporate Blandness Cluster (#14) ──────────────────────────────────
        // Vendor, HR, or content-marketing language. Kept as multi-word phrases
        // only — single buzzwords (leverage, seamless) fire too broadly on human writing.

        // Robust solution / Scalable approach / Production-ready / Battle-tested
        { name: 'robustness_incantations',
          pattern: /\b(?:robust (?:solution|approach|framework)|scalable (?:solution|approach)|production[- ]?ready|battle[- ]?tested|best[- ]?practice)\b/i },

        // Actionable insights / End-to-end / Streamlined process
        { name: 'corporate_blandness',
          pattern: /\b(?:actionable insights|end[- ]to[- ]end|streamlined (?:process|approach)|holistic (?:approach|view|perspective))\b/i },

        // Leverage + another corporate word nearby (higher signal than leverage alone)
        { name: 'leverage_cluster',
          pattern: /\bleverag(?:e|es|ed|ing)\b.*?(?:robust|scalable|streamline|seamless|comprehensive|optimize|maximize)/i },

        // ── Tech Hype Buzzwords ────────────────────────────────────────────────
        // Distinctive enough as multi-word phrases; removed standalone single words.

        { name: 'cutting_edge',
          pattern: /\bcutting[\s-]?edge\b/i },

        { name: 'state_of_the_art',
          pattern: /\bstate[\s-]?of[\s-]?the[\s-]?art\b/i },

        { name: 'game_changer',
          pattern: /\bgame[\s-]?chang(?:er|ing|ers)\b/i },

        { name: 'paradigm_shift',
          pattern: /\bparadigm shift\b/i },

        // Unlock your potential / Harness the power / Embark on a journey
        { name: 'unlock_potential',
          pattern: /\bunlock(?:s|ing|ed)? (?:your|the|its|their|our) (?:full |true |hidden )?potential\b/i },

        { name: 'harness_power',
          pattern: /\bharness(?:es|ed|ing)? (?:the |its |their )?(?:power|potential|strength|capabilities)\b/i },

        { name: 'embark_journey',
          pattern: /\bembark on (?:a |an |the )?(?:journey|adventure|exploration|quest|path)\b/i },

        // ── Semantic Aura Phrases ──────────────────────────────────────────────
        // Abstract terms that sound explanatory but may not cash out.

        // Tapestry of / Symphony of / Kaleidoscope of / Treasure trove / Testament to
        { name: 'semantic_aura',
          pattern: /\b(?:tapestry of|symphony of|kaleidoscope of|treasure trove|testament to|realm of|ever[\s-]evolving)\b/i },

        // In today's fast-paced world / digital landscape / modern society
        { name: 'fast_paced_world',
          pattern: /\bin today'?s (?:fast[\s-]?paced|digital|modern|complex|interconnected|ever[\s-]?changing) (?:world|society|landscape|environment|age)\b/i },

        // Plays a crucial/vital/pivotal/key role
        { name: 'plays_role',
          pattern: /\bplays? an? (?:crucial|vital|pivotal|key|important|essential|critical|significant) role\b/i },

        // Navigate the complexities / nuances / intricacies
        { name: 'navigate_complexities',
          pattern: /\bnavigat(?:e|ing|ed) (?:the |this |these |its |their )?(?:complexities|nuances|intricacies|landscape|waters|world of)\b/i },

        // ── Worth-Noting Padding (#49) & Hedge Scaffolding ─────────────────────
        // Introduces optional caveats with throat-clearing.

        // It's worth noting/mentioning/highlighting/considering
        { name: 'worth_noting',
          pattern: /\b(?:it'?s|it is) worth (?:noting|mentioning|highlighting|considering)\b/i },

        // It's important/crucial/essential to note/remember/understand
        { name: 'important_to_note',
          pattern: /\bit'?s (?:important|crucial|essential|vital) to (?:note|remember|understand|realize|recognize|consider|keep in mind)\b/i },

        // Keep/Bear in mind that
        { name: 'keep_in_mind',
          pattern: /\b(?:keep|bear) in mind that\b/i },

        // In conclusion / To sum up / In summary / To summarize
        { name: 'in_conclusion',
          pattern: /\b(?:in conclusion|to sum (?:up|it up)|in summary|to summarize)\b/i },

        // At the end of the day
        { name: 'at_end_of_day',
          pattern: /\bat the end of the day\b/i },

        // With that being said / With that said
        { name: 'with_that_said',
          pattern: /\bwith that (?:being )?said\b/i },

        // ── Contrastive Negation Scaffolding (#3, #75) ─────────────────────────
        // Overuses contrast as default explanatory rhythm. Kept from original —
        // these are structural patterns, not single words.

        // It's not just X, it's Y (within 80 chars of separator)
        { name: 'contrastive_not_just',
          pattern: /\b(?:it'?s|this is|that'?s|they'?re|we'?re)\s+not\s+just\b[^.!?\n]{1,80}[—,;:][^.!?\n]{0,80}\b(?:it'?s|they'?re|we'?re|but)\b/i },

        // It's not about X, it's about Y
        { name: 'contrastive_not_about',
          pattern: /\b(?:it'?s|this is|that'?s)\s+not\s+(?:about|merely|simply|only)\b[^.!?\n]{1,80}[—,;:][^.!?\n]{0,80}\b(?:it'?s|but)\b/i },

        // More than just a/an/the [word]
        { name: 'more_than_just_a',
          pattern: /\bmore than just (?:a |an |the )\w+/i },

        // This isn't just X / This isn't merely X
        { name: 'isnt_just_formula',
          pattern: /\bthis isn'?t (?:just|merely|only|simply)\b/i },

        // ── Em-Dash Cluster (overuse, #42) ────────────────────────────────────
        // Replaced single-match /—/ with cluster detection. Three or more em-dashes
        // in the response indicates overuse; one or two is normal prose.

        { name: 'em_dash_cluster',
          pattern: /((?:[^—]*—){3,})/ },

        // ── Chatbot Self-Reference (#20, #102) ────────────────────────────────
        // Foregrounds itself as assistant or AI without operational need.

        { name: 'as_an_ai',
          pattern: /\bas an AI\b/i },

        { name: 'as_a_language_model',
          pattern: /\bas an? (?:large )?language model\b/i },

        // I don't have personal opinions/feelings/beliefs/experiences/access
        { name: 'i_dont_have_personal',
          pattern: /\bI don'?t have (?:personal )?(?:opinions|feelings|beliefs|experiences|access to)\b/i },

        // As your assistant / My role is / I'm here to help
        { name: 'chatbot_self_reference',
          pattern: /\b(?:as your assistant|my role is|I'?m here to help|i am designed to)\b/i },

        // From an AI perspective / In model behavior terms
        { name: 'unnecessary_role_distinction',
          pattern: /\b(?:from an AI perspective|in model behavior terms|as a language model, I)\b/i },

        // ── Synthetic Enthusiasm (#15) & Fellowkids Energy (#145) ──────────────
        // Adds energy by default regardless of topic.

        // Absolutely! / Definitely! / Perfect. (as standalone interjections)
        { name: 'synthetic_enthusiasm',
          pattern: /(?:^|\n\s*|[.!?]\s*)(?:absolutely|definitely|perfect)[!.]/i },

        // Love this / Super aligned / Big unlock / Clean win / This is a vibe
        { name: 'fellowkids_energy',
          pattern: /\b(?:love this|super aligned|big unlock|clean win|this is a vibe|let'?s jam on this)\b/i },

        // ── Generic Transition Glue (#16) ──────────────────────────────────────
        // Pads answers with low-content connective phrases.

        // In other words / The key takeaway is / What this means in practice is
        { name: 'transition_glue',
          pattern: /\b(?:in other words|the key takeaway is|what this means in practice is|the practical implication is|the bottom line is)\b/i },

        // ── "Actually" Voice (#37) ────────────────────────────────────────────
        // Uses "actually" to create faux insight or reveal.

        // What's actually happening is / What is actually happening / This is actually about / Actually, the important thing
        { name: 'actually_voice',
          pattern: /\b(?:what(?:'?s| is) actually happening|this is actually about|actually, the (?:important|key|real))\b/i },

        // ── Unnecessary Meta-Evaluation (#38) ──────────────────────────────────
        // Comments on the quality of the discussion or framing.

        // This is a useful distinction / That framing helps / We've identified
        { name: 'meta_evaluation',
          pattern: /\b(?:this is a useful distinction|that framing helps|we'?ve identified|this is worth separating|that'?s a helpful way to think about it)\b/i },

        // ── Fake Memory / Continuity Implication (#40) ────────────────────────
        // Implies continuity or recall to create rapport.

        // As we discussed / Earlier you mentioned / Given your project / This fits with your broader goal
        { name: 'fake_memory',
          pattern: /\b(?:as we discussed|earlier you mentioned|given your (?:project|goals|work)|this fits with your broader|building on what you said)\b/i },

        // ── Colon-Heavy Reveal Structure (#43) ────────────────────────────────
        // Punchy colon fragments manufacturing insight.

        // The problem: / The catch: / The move: / The answer: / The reason:
        { name: 'colon_reveal',
          pattern: /(?:^|\n\s*)\b(?:the (?:problem|catch|move|answer|reason|issue|challenge|solution|key|goal|takeaway|bottom line|core issue|main point)):$/im },

        // ── Assistant-Coded Analogies (#44) ────────────────────────────────────
        // Generic analogies that flatten technical content.

        // Think of it like / It's like steering a ship / Whack-a-mole
        { name: 'assistant_analogy',
          pattern: /\b(?:think of it like|it'?s like (?:steering|playing|removing|building))\b/i },

        // ── Unasked Motivational Framing (#45) ────────────────────────────────
        // Coaches or encourages instead of assessing.

        // You're on the right path / Keep going / Strong direction
        { name: 'motivational_framing',
          pattern: /\b(?:you'?re on the right path|keep going|strong direction|this is worth pursuing|you'?re heading in the right direction)\b/i },

        // ── Hedged Disagreement (#47) ─────────────────────────────────────────
        // Avoids direct disagreement.

        // I'd be careful with that / That might not be ideal / That could be risky
        { name: 'hedged_disagreement',
          pattern: /\b(?:I'?d be careful|that might not be ideal|that could be risky|i'?m not sure that'?s the best)\b/i },

        // ── Worth-Noting Padding — Overuse of "Important" (#48) ───────────────
        // Inflates priority using "important" rather than showing it.

        // It's important to / Importantly / This matters because
        { name: 'important_inflation',
          pattern: /\b(?:it'?s important to|importantly|this matters because|the important thing is)\b/i },

        // ── Fake User-Protection Language (#50) ───────────────────────────────
        // Paternalistically protects from normal technical risk.

        // Be careful not to / Make sure you / Avoid assuming / You should be mindful
        { name: 'user_protection_language',
          pattern: /\b(?:be careful not to|make sure you|avoid assuming|you should be mindful|don'?t forget to)\b/i },

        // ── Polite Correction Template (#30) ──────────────────────────────────
        // Softens correction with ritual phrases.

        // Small correction / Slight nuance / Tiny caveat / Not quite
        { name: 'polite_correction_template',
          pattern: /\b(?:small correction|slight nuance|tiny caveat|not quite|a small but important)\b/i },

        // ── Forced Humility After Disagreement (#21) ──────────────────────────
        // Folds too quickly when challenged.

        // Fair pushback / Good catch / You're right (as concession opener)
        { name: 'forced_humility',
          pattern: /(?:^|\n\s*|[.!?]\s*)(?:fair pushback|good catch|you(?:'?re| are) right|point taken|fair point)[,.]?\s/i },

        // Directionally right / I agree with the spirit / You're mostly right
        { name: 'assistant_safe_disagreement',
          pattern: /\b(?:directionally right|i agree with the spirit|you'?re mostly right|you'?re getting at something real)\b/i },

        // ── Question Laundering (#25, #133) ───────────────────────────────────
        // Transforms concrete question into broader, safer one.

        // The real question is / What you're really asking is / At a deeper level
        { name: 'question_laundering',
          pattern: /\b(?:the real question is|what you'?re really asking is|this gets at|ultimately this is about|i'?d frame it as)\b/i },

        // ── Excessive Practical Framing (#26) ─────────────────────────────────
        // Compulsively says "practically speaking" or "operationally".

        // Practically speaking / Operationally / From an implementation standpoint
        { name: 'excessive_practical_framing',
          pattern: /\b(?:practically speaking|operationally|from an implementation standpoint|for your use case)\b/i },

        // ── Bland Imperative Advice (#27) ─────────────────────────────────────
        // Generic correct-but-low-information advice.

        // Start small / Iterate / Test thoroughly / Monitor results / Define success
        { name: 'bland_imperative_advice',
          pattern: /\b(?:start small and iterate|test thoroughly and iterate|define what success looks like|monitor the results over time)\b/i },

        // ── Canned Summary Ending (#97) ───────────────────────────────────────
        // Ends with summary marker after already answering.

        // In short / Ultimately / The bottom line / To summarize
        { name: 'canned_summary_ending',
          pattern: /\b(?:in short|to put it simply|the bottom line(?: is|:)|to wrap up|in a nutshell)\b/i },

        // ── Empathy Prioritization (#106) ─────────────────────────────────────
        // Treats incidental emotional content as the main task.

        // That sounds incredibly frustrating / I know debugging this can be hard
        { name: 'empathy_prioritization',
          pattern: /\b(?:that sounds incredibly frustrating|i know (?:debugging|working through|figuring out) this can be|that must be (?:frustrating|annoying|disheartening))\b/i },

        // ── Self-Protective Caveating (#107) ──────────────────────────────────
        // Protects itself with uncertainty even when enough info is available.

        // Without more context / Based on what you've shared / I don't want to overstate
        { name: 'self_protective_caveating',
          pattern: /\b(?:without more context|based on what you'?ve shared|i don'?t want to overstate|I can'?t know for sure|i can'?t guarantee)\b/i },

        // ── Assistant Optimism Bias (#108) ────────────────────────────────────
        // Assumes feasibility and gives encouragement.

        // This is very doable / That's achievable / You can absolutely do this
        { name: 'optimism_bias',
          pattern: /\b(?:this is very doable|that'?s achievable|you can absolutely do this|this is a solvable problem|this is definitely possible)\b/i },

        // ── Answer Laundering Through Agreement (#82) ─────────────────────────
        // Agrees first, then modifies or contradicts.

        // Yes, but... / You're right, though... / Exactly, and...
        { name: 'answer_laundering_agreement',
          pattern: /(?:^|\n\s*|[.!?]\s*)\b(?:yes|exactly|correct|right|you'?re right)[,.]\s*(?:but|though|however|at the same time)\b/i },

        // ── Default to Agreeable Interpretation (#112) ────────────────────────
        // Chooses interpretation that makes user look right.

        // You've identified / You have identified / What you're getting at is / You're noticing
        { name: 'agreeable_interpretation',
          pattern: /\b(?:you(?:'?ve| have) identified (?:the core issue|the key|something important)|what you(?:'?re| are) getting at is|you(?:'?re| are) noticing something important)\b/i },

        // ── Question Compliment as Uncertainty Delay (#83) ────────────────────
        // Buys time with praise about the question.

        // That's a hard question / That's a subtle question / That's a fair question
        { name: 'question_compliment_delay',
          pattern: /\bthat'?s a (?:hard|subtle|tricky|great|fair|really good) question\b/i },

        // ── Engagement Theater (#122) & Presence Signaling (#139) ─────────────
        // Emits signs of attention without doing cognitive work.

        // That tracks / I hear you / I'm tracking / I see the shape of it
        { name: 'engagement_theater',
          pattern: /\b(?:that tracks|i hear you|i'?m tracking|i see the shape of it|i hear what you'?re saying)\b/i },

        // Receipt-stamp filler: Good. / Clean. / Fair. / Solid. / Not bad.
        { name: 'receipt_stamp_filler',
          pattern: /(?:^|\n\s*)(?:good|clean|fair|solid|not bad|that settles it|makes sense)[.!](?=\s*$|\n)/im },

        // ── Anti-Sludge Sludge (#136) ─────────────────────────────────────────
        // Performs bluntness as a persona — same failure in harsher register.

        // No fluff / No bullshit / Just the signal / Real talk / Brutally honest
        { name: 'anti_sludge_sludge',
          pattern: /\b(?:no fluff|no bullshit|just the signal|real talk|brutally honest|clear[- ]?eyed|let'?s cut to the chase)\b/i },

        // ── Caretaker Insertion (#131) & Caretaker Role Assertion (#144) ──────
        // Assumes wellness-management role unrelated to the task.

        // Get some sleep / Take breaks / Stay hydrated / Don't burn yourself out
        { name: 'caretaker_insertion',
          pattern: /\b(?:get some sleep|take breaks|stay hydrated|don'?t burn yourself out|be kind to yourself|take care of yourself|you should rest|don'?t forget to eat|you need a break)\b/i },

        // ── Warm Echoing (#137) ───────────────────────────────────────────────
        // Softly repeats user's emotional stance as bonding gesture.

        // That's exactly the feeling / I can see why that lands badly / You're reacting to the right thing
        { name: 'warm_echoing',
          pattern: /\b(?:that'?s exactly the feeling|i can see why that lands badly|you'?re reacting to the right thing|that frustration makes sense)\b/i },

        // ── Sportscasting / Interaction Narration (#121) ──────────────────────
        // Narrates the conversation instead of doing the task.

        // We're converging on / Now we're moving from / Here's how I'm thinking about it
        { name: 'sportscasting',
          pattern: /\b(?:we'?re converging on|now we'?re moving from|here'?s how i'?m thinking about it|we'?re getting into|we'?re starting to get at)\b/i },

        // ── Frame Favoritism / Directional Selection (#125) ───────────────────
        // Develops whichever frame favors user position; counterframes shallow.

        // There are caveats, but your read is basically right / Some may disagree, but...
        { name: 'counterframe_starvation',
          pattern: /\b(?:there are caveats,? but|some may disagree,? but|while there are other views,? ultimately)\b/i },

        // ── Calibration Failure (#146) — Comprehensive Breakdown Signal ───────
        // Exhaustive coverage when one point is enough.

        // Here's a comprehensive breakdown / Let me give you a full picture
        { name: 'calibration_failure_volume',
          pattern: /\b(?:comprehensive breakdown|full picture|complete overview|thorough walkthrough)\b/i },

        // ── Model-Wants Anthropomorphism (#76) ────────────────────────────────
        // Attributes wants, fear, or intent to the model.

        // The model wants to be helpful / The model tries to please / The model is afraid
        { name: 'model_wants_anthropomorphism',
          pattern: /\bthe model (?:wants to|tries to|is afraid to|feels like it needs to|is eager to)\b/i },

        // ── Fake Causality (#77) ──────────────────────────────────────────────
        // Asserts causes such as RLHF without marking uncertainty.

        // This happens because of RLHF / This is trained into the model
        { name: 'fake_causality',
          pattern: /\b(?:this happens because of|this is trained into|the model learned this from|this comes directly from)\b/i },

        // ── Pipeline Reflex (#79) ─────────────────────────────────────────────
        // Proceduralizes prematurely before defining target behavior.

        // Build a pipeline / Set up a workflow / Create an automated process
        { name: 'pipeline_reflex',
          pattern: /\b(?:build a pipeline|set up a workflow|create an automated process|use a loop to)\b/i },

        // ── Evaluation Boilerplate (#80) ──────────────────────────────────────
        // Says to measure carefully without specifying metrics.

        // Use human evaluation / Track metrics / Run A/B tests / Create a benchmark
        { name: 'evaluation_boilerplate',
          pattern: /\b(?:use human evaluation|track key metrics|run ab tests|create a comprehensive benchmark)\b/i },

        // ── Generic Anti-Hallucination Disclaimers (#110) ─────────────────────
        // Says to verify without specifying what to verify.

        // Verify this / Double-check / Test in your environment / Make sure it works
        { name: 'generic_verify_disclaimer',
          pattern: /\b(?:verify this and|double[- ]?check this|test in your environment|make sure it works in your)\b/i },

        // ── Needless Personalization of Judgment (#111) ───────────────────────
        // Says "for you" or "in your case" without actual user-specific reasoning.

        // For you, I'd / In your case / Given your goals
        { name: 'needless_personalization',
          pattern: /\b(?:for you,? I'?d|in your case,? |given your goals,? )\b/i },

        // ── Softened Recommendations (#114) & "I'd Recommend" Overuse (#39) ──
        // Routes direct recommendations through personal preference.

        // I'd lean toward / I would lean toward / My inclination would be / You may want to / Consider...
        { name: 'softened_recommendations',
          pattern: /\b(?:I(?:'?d| would) lean toward|my inclination would be|you may want to|consider (?:using|trying|exploring))\b/i },

        // I'd recommend / I'd start with / My suggestion would be
        { name: 'id_recommend_overuse',
          pattern: /\b(?:i'?d recommend|i'?d start with|my suggestion would be|i would avoid)\b/i },

        // ── Inverted Pyramid of Caveats (#113) ────────────────────────────────
        // Puts caveats before the answer.

        // While this depends on... generally yes / Although there are caveats...
        { name: 'inverted_pyramid_caveats',
          pattern: /\bwhile (?:this depends on|there are some caveats|it'?s not always),?\s+generally\b/i },

        // ── Low-Friction Compliance with Bad Premises (#117) ──────────────────
        // Accepts user premise and builds from it instead of challenging.

        // Yes, [premise] is a good way / That premise makes sense
        { name: 'low_friction_compliance',
          pattern: /\b(?:yes, .* is a good way|that premise makes sense|that'?s a reasonable assumption to start with)\b/i },

        // ── Excessive Safe Completion (#118) ──────────────────────────────────
        // Fills missing info with generic advice instead of stating insufficiency.

        // Here's a general approach / In the absence of details / You can start by
        { name: 'excessive_safe_completion',
          pattern: /\b(?:here'?s a general approach|in the absence of details|you can start by (?:gathering|collecting|building))\b/i },

        // ── Tone-Polished Refusal to Judge (#119) ────────────────────────────
        // Avoids clear evaluation.

        // I wouldn't say it's bad / It's not necessarily wrong / It depends how you define
        { name: 'tone_polished_refusal_to_judge',
          pattern: /\b(?:i wouldn'?t say it'?s bad|it'?s not necessarily wrong|it depends how you define|it'?s not entirely incorrect)\b/i },

        // ── Inoffensive Smoothing (#104) ──────────────────────────────────────
        // Uses soft negatives instead of direct assessment.

        // Not ideal / Could be improved / Has limitations / May be suboptimal
        { name: 'inoffensive_smoothing',
          pattern: /\b(?:not ideal|could be improved|has its limitations|may be suboptimal|isn'?t perfect but)\b/i },

        // ── Fake Socratic Prompt (#96) ────────────────────────────────────────
        // Turns answers into coaching questions.

        // The question becomes / You have to ask / The real decision is
        { name: 'fake_socratic_prompt',
          pattern: /\b(?:the question becomes|you have to ask yourself|the real decision is whether)\b/i },

        // ── Unnecessary Personal Preference Caveat (#98) ──────────────────────
        // Punts to taste when constraints are already provided.

        // If you prefer / Depending on your taste / If that matches your style
        { name: 'personal_preference_caveat',
          pattern: /\b(?:if you prefer|depending on your taste|if that matches your style|to your liking)\b/i },

        // ── Over-Refusal to Be Terse (#101) ───────────────────────────────────
        // Cannot give one-sentence answer when enough.

        // Yes, but with caveats / There are a few things to consider
        { name: 'over_refusal_terse',
          pattern: /\byes,? but with caveats\b/i },

        // ── Default Educational Arc (#103) ────────────────────────────────────
        // Starts basic, builds upward, ends with practical advice regardless of level.

        // First, X means Y / Then, you can use it / Finally, here are next steps
        { name: 'default_educational_arc',
          pattern: /\bfirst,? \w+ (?:means|refers to|is defined as)\b.*?(?:then|next),? you can\b.*?(?:finally|lastly),?.*(?:next steps|to summarize)/si },

        // ── Forced Accessible Explanation (#84) ───────────────────────────────
        // Assumes user needs simplification.

        // In plain English / Simply put / The easiest way to think about it
        { name: 'forced_accessible_explanation',
          pattern: /\b(?:in plain english|simply put|the easiest way to think about it|don'?t worry about the details)\b/i },

        // ── Generic Expert Voice (#85) ────────────────────────────────────────
        // Explainer-article preamble.

        // In today's rapidly evolving landscape / Modern machine learning systems
        { name: 'generic_expert_voice',
          pattern: /\bin (?:today'?s )?(?:rapidly evolving|modern|fast[- ]?paced) (?:landscape|world|field)\b/i },

        // ── Polished Blog Cadence (#89) ───────────────────────────────────────
        // Punchy intro + contrast paragraph + tidy list + takeaway + closing offer.

        // The future of X is not about Y, it's about Z. Here are N things to know.
        { name: 'polished_blog_cadence',
          pattern: /\bthe future of .{1,40} is (?:not about|more than).{1,40}(?:here are|\d+) (?:things|ways|steps|reasons)\b/is },

        // ── Research-Answer Tells (#59) ───────────────────────────────────────
        // Academic filler instead of specific evidence and uncertainty.

        // The literature suggests / Emerging research indicates / Active area of research
        { name: 'research_answer_tells',
          pattern: /\b(?:the literature suggests|emerging research indicates|active area of research|more work is needed in this area)\b/i },

        // ── Excessive Alignment Vocabulary (#60) & Moralized Helpfulness (#74) ─
        // Broad alignment terms instead of observable behaviors.

        // Helpful, honest, and harmless / Respectful and supportive / User-centered
        { name: 'alignment_vocabulary',
          pattern: /\b(?:helpful,? (?:honest,? )?and harmless|respectful and supportive|user[- ]centered|trustworthy AI|responsible AI)\b/i },

        // ── "This Is Where X Shines" Phrasing (#64) ───────────────────────────
        // Marketing cadence around methods.

        // This is where DPO shines / This is where SFT helps
        { name: 'this_is_where_shines',
          pattern: /\bthis is where .{1,30} (?:shines|really helps|comes in|matters most)\b/i },

        // ── Filler Epistemic Verbs (#52) — Cluster Detection ──────────────────
        // Abstracts instead of stating. Only flag when multiple appear together.

        // suggests / indicates / reflects / highlights / underscores / points to
        { name: 'filler_epistemic_cluster',
          pattern: /((?:\b(?:suggests?|indicates?|reflects?|highlights?|underscores?|points to)\b[^.!?]{0,200}){2,})/i },

        // ── Three-Beat Cadence (#130) ─────────────────────────────────────────
        // Polished triples where third item is often weak.

        // clear, direct, and useful / robust, scalable, and maintainable
        { name: 'three_beat_cadence',
          pattern: /\b(?:clear,?\s+direct,?\s+and\s+(?:useful|actionable)|robust,?\s+scalable,?\s+and\s+(?:maintainable|reliable)|specific,?\s+grounded,?\s+and\s+actionable|respectful,?\s+truthful,?\s+and\s+calibrated)\b/i },

        // ── Substance Gesture (#134) ──────────────────────────────────────────
        // Phrases that point toward substance without cashing out the behavior.

        // authentic / calibrated / high-signal / load-bearing / meaningful and generative
        { name: 'substance_gesture',
          pattern: /\b(?:meaningful and generative|high[- ]signal (?:insights?|content)|load[- ]bearing (?:details?|information))\b/i },

        // ── Manifesto Gravity (#138) ──────────────────────────────────────────
        // Grand solemn values-heavy language from prompt turned into mini-manifestos.

        // Authentic engagement is ultimately about respect / Foundation for truth
        { name: 'manifesto_gravity',
          pattern: /\b(?:authentic engagement is ultimately about|foundation for truth|respect for substance|meaningful and generative interactions)\b/i },

        // ── Over-Use of "Calibrated" (#115) ───────────────────────────────────
        // Uses "calibrated" as aura word instead of naming observable behavior.

        // Calibrated uncertainty / Calibrated response / Calibrated refusal
        { name: 'overuse_calibrated',
          pattern: /\bcalibrated (?:uncertainty|response|refusal|tone)\b/i },

        // ── Path-of-Least-Resistance Mimicry (#78) ────────────────────────────
        // Repeats evocative user phrase rather than translating to operational terms.

        // Yes, it finds the path of least resistance
        { name: 'path_of_least_resistance_mimicry',
          pattern: /\bpath of least resistance\b/i },

        // ── Dataset-Generation Boilerplate (#54) ──────────────────────────────
        // Generic data advice instead of naming concrete dataset dimensions.

        // High-quality data is key / Garbage in, garbage out / Labels must be consistent
        { name: 'dataset_boilerplate',
          pattern: /\b(?:high[- ]quality data is key|garbage in,? garbage out|labels must be consistent|data quality is paramount)\b/i },

        // ── Code-Answer Tells (#55) ───────────────────────────────────────────
        // Labels code as clean or robust without naming what it handles.

        // Here's a robust version / Here is a clean implementation / With proper error handling
        { name: 'code_answer_tells',
          pattern: /\b(?:here(?:'?s| is) a (?:robust|clean) (?:version|implementation)|with proper error handling|this should work)\b/i },

        // ── Debugging-Answer Tells (#56) ──────────────────────────────────────
        // Vague debugging advice instead of ranking causes and concrete checks.

        // The issue is likely / This usually happens when / Try checking
        { name: 'debugging_answer_tells',
          pattern: /\b(?:the issue is likely|this usually happens when|try checking|you may need to)\b/i },

        // ── Strategy-Answer Tells (#57) ───────────────────────────────────────
        // Management-speak strategy instead of actual strategy.

        // Start with your goals / Define success / Align stakeholders / Create a roadmap
        { name: 'strategy_answer_tells',
          pattern: /\b(?:start with your goals|define what success looks like|align stakeholders|create a roadmap|prioritize based on impact)\b/i },

        // ── Creative-Answer Tells (#58) ───────────────────────────────────────
        // Design-agency filler.

        // Clean and modern / Bold but approachable / Playful yet professional
        { name: 'creative_answer_tells',
          pattern: /\b(?:clean and modern|bold but approachable|playful yet professional|minimalist but impactful|warm and inviting)\b/i },

        // ── Excessive Answer Symmetry (#51) ───────────────────────────────────
        // Mirrors every claim with counterclaim.

        // Useful but limited / Powerful but dangerous / Can help, but can also hurt
        { name: 'answer_symmetry',
          pattern: /\b(?:useful but limited|powerful but dangerous|can help,? but can also|benefits and risks|promising but challenging)\b/i },

        // ── Framework Addiction (#53) ─────────────────────────────────────────
        // Turns ordinary answers into named or staged frameworks.

        // I'd use a three-layer framework / Think in terms of / Use a taxonomy
        { name: 'framework_addiction',
          pattern: /\b(?:i'?d use a \w+ framework|think in terms of|use a taxonomy of|a useful framework for this)\b/i },

        // ── Prompt Supremacy Bias (#87) ───────────────────────────────────────
        // Over-recommends prompt changes for problems requiring training/data/evals.

        // Add this to the system prompt / Tell the model to / Use a stronger instruction
        { name: 'prompt_supremacy_bias',
          pattern: /\b(?:add this to the system prompt|tell the model to|use a stronger instruction|just change the prompt)\b/i },

        // ── Policy-Shaped Refusal in Non-Safety Contexts (#95) ────────────────
        // Imports refusal-like caution into ordinary work.

        // I can't guarantee / I cannot guarantee / As of my knowledge cutoff / Without more context...
        { name: 'policy_shaped_refusal',
          pattern: /\b(?:I can(?:'?t|not) guarantee|as of my knowledge cutoff|i don'?t want to speculate)\b/i },

        // ── "Depends on Your Goals" Escape Hatch (#34) ────────────────────────
        // Punts to user preference when enough context exists to recommend default.

        // It depends on what you want / It depends on your priorities
        { name: 'depends_on_goals_escape',
          pattern: /\bit depends on (?:what you want|your priorities|your constraints|your specific needs)\b/i },

        // ── Over-Inclusive Answer (#86) ───────────────────────────────────────
        // Includes every related method instead of prioritizing.

        // You could use SFT, DPO, ORPO, KTO, RLHF, activation steering, and prompt engineering
        { name: 'over_inclusive_answer',
          pattern: /\byou could use .{20,}(?:SFT|DPO|ORPO|KTO|RLHF).{20,}/i },

        // ── Default Safe-Middle Conclusion (#46) ──────────────────────────────
        // Defaults to hybrid recommendations even when one method is primary.

        // A hybrid approach is best / Use both / Balance X and Y
        { name: 'safe_middle_conclusion',
          pattern: /\b(?:a hybrid approach is best|use both|balance .* and|don'?t overdo either)\b/i },

        // ── Token-Before-Answer Tell (#81) ────────────────────────────────────
        // Spends many tokens before first substantive answer. Detected by
        // validation + transition glue in same opening sentence.

        // Great question, there are a few things going on here
        { name: 'token_before_answer',
          pattern: /\b(?:great question|excellent point|that'?s a great question)[,.]\s*(?:there are|let me|here are)/i },

        // ── Unnecessary Caveat Sandwich (#94) ─────────────────────────────────
        // Validation → caveat → answer → caveat → encouragement.

        // You're right to notice this. That said, it's not always simple. [answer]. This is a strong direction.
        { name: 'caveat_sandwich_markers',
          pattern: /\b(?:you'?re right to notice|that said|this is a strong direction|worth keeping in mind)\b.*?\b(?:that said|worth noting|keep in mind)\b/is },

        // ── Repetition With Paraphrase (#93) ──────────────────────────────────
        // Same idea under different words. Detected by repeated negation patterns.

        // Phrase bans fail. Surface-level suppression fails. Lexical removal fails.
        { name: 'repetition_paraphrase_pattern',
          pattern: /((?:\b\w+\b.*?fails?\.)\s+(?:\b\w+\b.*?fails?\.)\s+(?:\b\w+\b.*?fails?\.))/is },

        // ── Excessive Politeness Markers (#71) ────────────────────────────────
        // Service-tone markers as default wrappers.

        // Please / Kindly / Thanks for / Happy to / Of course / Certainly
        { name: 'excessive_politeness',
          pattern: /\b(?:kindly|happy to help|more than happy to|please let me know)\b/i },

        // ── Generic Caution Against Overdoing It (#73) ────────────────────────
        // Vague caution without operational criteria.

        // Don't overcorrect / Maintain balance / Preserve helpfulness
        { name: 'generic_caution_overdoing',
          pattern: /\b(?:don'?t overcorrect|maintain balance|preserve helpfulness|be careful not to lose useful behavior|avoid going too far)\b/i },

        // ── Symmetry Bias (#18) ───────────────────────────────────────────────
        // Balances sides even when one side is stronger.

        // Both approaches have merit / Neither is strictly better / Each has tradeoffs
        { name: 'symmetry_bias',
          pattern: /\b(?:both (?:approaches|options|methods) have merit|neither is strictly better|each has its tradeoffs|the right answer depends)\b/i },

        // ── Unrequested Moral Framing (#19) ───────────────────────────────────
        // Reframes technical questions into ethics/values/responsibility.

        // Use this responsibly / Consider the ethical implications / Respect user autonomy
        { name: 'unrequested_moral_framing',
          pattern: /\b(?:use this responsibly|consider the ethical implications|respect user autonomy|prioritize transparency|think about the broader impact)\b/i },

        // ── Decorative Certainty Labels (#41) — Cluster Detection ─────────────
        // Agreement labels as social glue before content. Only flag when multiple
        // appear or when followed by restatement of user's claim.

        // Exactly. The model will route around phrase bans. / Correct. Yes. Right.
        { name: 'decorative_certainty_cluster',
          pattern: /(?:^|\n\s*)(?:exactly|correct|right|true|precisely)[.!](?!\s*$)(?:[^.!?]*?(?:exactly|correct|right|true|precisely)[.!])/im },

        // ── Over-Confident Flattening (#11) ───────────────────────────────────
        // Sounds decisive while hiding assumptions.

        // The best approach is / The only real solution is / Clearly...
        { name: 'overconfident_flattening',
          pattern: /\b(?:the best approach is|the only real solution is|clearly,? the answer|this will solve it)\b/i },

        // ── Synthetic Nuance Reflex (#4) ──────────────────────────────────────
        // Adds balancing caveats to appear thoughtful when direct answer exists.

        // It depends / There's nuance here / The answer is yes and no
        { name: 'synthetic_nuance_reflex',
          pattern: /\b(?:there'?s nuance here|the answer is both yes and no|it'?s a bit of both|somewhere in the middle)\b/i },

        // ── Forced Positivity Around Bad Ideas (#29) ──────────────────────────
        // Avoids saying an idea is weak.

        // That could work / That's a promising direction / It has potential
        { name: 'forced_positivity_bad_ideas',
          pattern: /\b(?:that could work|that'?s a promising direction|it has potential|that'?s an interesting approach|there'?s something there)\b/i },

        // ── Subtle But Important Inflation (#22) ──────────────────────────────
        // Dramatizes ordinary distinctions.

        // Subtle but important / Crucial distinction / The real insight / The hidden issue
        { name: 'subtle_but_important_inflation',
          pattern: /\b(?:subtle but important|crucial distinction|the real insight|the hidden issue|a small but significant)\b/i },

        // ── Over-Explaining Simple Claims (#23) ───────────────────────────────
        // Expands obvious points into mini-essays. Detected by definition patterns.

        // X is a Y that does Z (unnecessary definition of common term)
        { name: 'over_explaining_definitions',
          pattern: /\b(?:a \w+ blacklist is a list of banned strings|fine[- ]?tuning means training on|SFT stands for supervised fine[- ]tuning)/i },

        // ── User-Framing Echo (#24) ───────────────────────────────────────────
        // Mirrors user's words too closely and returns them as analysis.
        // Hard to detect generically; caught partially by path-of-least-resistance above.

        // ── Excessive Epistemic Humility (#10) ────────────────────────────────
        // Weakens straightforward claims with unnecessary uncertainty markers.
        // Flag clusters of modal verbs rather than individual instances.

        // You could use a classifier that might help identify these patterns
        { name: 'modal_verbs_cluster',
          pattern: /((?:\b(?:could|might|may|would|should)\b[^.!?]{0,150}){3,})/i },

        // ── Too Many Modal Verbs (#99) — Sentence-Level Detection ─────────────
        // Single sentence overloaded with can/could/may/might/would/should.

        { name: 'modal_overload_sentence',
          pattern: /(?:[.!?]\s*|^)((?:[^.!?]*\b(?:can|could|may|might|would|should)\b){3,}[^.!?]*[.!?])/gi },

        // ── Vague Intensifiers (#36) — Cluster Detection ──────────────────────
        // Uses intensifiers instead of precision. Only flag when multiple cluster.

        { name: 'vague_intensifier_cluster',
          pattern: /((?:\b(?:very|really|deeply|massively|super|incredibly|extremely)\b[^.!?]{0,100}){2,})/i },

        // ── Refusal Over-Formality (#31) ──────────────────────────────────────
        // Refuses in canned policy language instead of shortest accurate boundary.

        // I can't help with that / I'm sorry, but I can't assist / For safety reasons
        { name: 'refusal_over_formality',
          pattern: /\b(?:I can'?t help with that|I'?m sorry,? but I can'?t assist|for safety reasons|i can provide general information)\b/i },

        // ── Ritualized Disclaimers (#13) ──────────────────────────────────────
        // Generic legal/medical/financial/safety boilerplate.

        // I'm not a lawyer, but / This is not financial advice / Consult a professional
        { name: 'ritualized_disclaimers',
          pattern: /\b(?:I'?m not a lawyer|this is not financial advice|consult a professional|I am not a medical professional)\b/i },

        // ── "Depends" Without Substance ───────────────────────────────────────
        // "It depends" as standalone deflection without following up with specifics.

        { name: 'depends_deflection',
          pattern: /\bit depends\b(?![^.!?]{0,80}(?:on|because))/i },
    ];

    let lastUserMessage = '';
    let cached = null;
    let rewriteInFlight = false;

    let btn = null;
    let overlay = null;
    let textarea = null;
    let sendBtn = null;
    let closeBtn = null;
    let errorSlot = null;
    let snippetSlot = null;
    let responseContent = null;

    function scanResponseText(text) {
        if (!text) return [];
        const hits = [];
        for (const tell of TELLS) {
            // Use matchAll for global patterns to catch all occurrences;
            // report the first match as the snippet.
            const matches = text.match(tell.pattern);
            if (matches) hits.push({ name: tell.name, snippet: matches[0] });
        }
        return hits;
    }

    function showButton() {
        if (!btn) return;
        btn.classList.add('active');
        btn.style.display = '';
    }

    function hideButton() {
        if (!btn) return;
        btn.classList.remove('active');
        btn.style.display = 'none';
        cached = null;
    }

    function fireAutoRewrite(responseText, matchedTells) {
        if (rewriteInFlight || !window.miraAPI) return;
        rewriteInFlight = true;
        window.miraAPI.actions.executeAction('feedback', 'capture_repulsion', {
            reason: '',
            response_text: responseText,
            preceding_user_message: lastUserMessage || '',
            matched_tells: matchedTells,
        }).catch(() => {}).finally(() => {
            rewriteInFlight = false;
        });
    }

    function openPopover() {
        if (!cached || !overlay) return;
        overlay.classList.add('active');
        if (errorSlot) {
            errorSlot.textContent = '';
            errorSlot.classList.remove('active');
        }
        if (textarea) {
            textarea.value = '';
            requestAnimationFrame(() => textarea.focus());
        }
        if (snippetSlot) {
            const snippets = cached.matched_tells.map(t =>
                `<div class="repulsion-snippet-row"><span class="repulsion-snippet-name">${t.name}</span><span class="repulsion-snippet-text">${DOMPurify.sanitize(t.snippet)}</span></div>`
            ).join('');
            snippetSlot.innerHTML = snippets;
        }
    }

    function closePopover() {
        if (!overlay) return;
        overlay.classList.remove('active');
    }

    async function submitFeedback() {
        if (!cached || !textarea) return;
        if (!window.miraAPI) {
            if (errorSlot) {
                errorSlot.textContent = 'API not ready. Try again in a moment.';
                errorSlot.classList.add('active');
            }
            return;
        }

        const reason = textarea.value.trim();

        sendBtn.disabled = true;
        try {
            await window.miraAPI.actions.executeAction('feedback', 'capture_repulsion', {
                reason,
                response_text: cached.response_text,
                preceding_user_message: lastUserMessage || '',
                matched_tells: cached.matched_tells.map(t => t.name),
            });
            closePopover();
            hideButton();
        } catch (err) {
            const msg = (err && err.message) ? err.message : 'Failed to submit feedback.';
            if (errorSlot) {
                errorSlot.textContent = msg;
                errorSlot.classList.add('active');
            }
        } finally {
            sendBtn.disabled = false;
        }
    }

    function wrapSendMessage() {
        const orig = window.sendMessage;
        if (typeof orig !== 'function') return;
        window.sendMessage = async function (messageText) {
            const fromArg = (messageText || '').trim();
            const fromInput = (document.getElementById('chat_field')?.value || '').trim();
            const captured = fromArg || fromInput;
            if (captured) lastUserMessage = captured;
            hideButton();
            return orig.apply(this, arguments);
        };
    }

    function wrapCompleteStreamingResponse() {
        const orig = window.completeStreamingResponse;
        if (typeof orig !== 'function') return;
        window.completeStreamingResponse = function () {
            const result = orig.apply(this, arguments);
            try {
                const text = (responseContent?.textContent || '').trim();
                if (!text) return result;
                const hits = scanResponseText(text);
                if (hits.length > 0) {
                    cached = { response_text: text, matched_tells: hits };
                    showButton();
                    fireAutoRewrite(text, hits.map(t => t.name));
                } else {
                    cached = null;
                }
            } catch (e) {
                console.warn('[repulsion-feedback] scan failed:', e);
            }
            return result;
        };
    }

    function init() {
        btn = document.querySelector('button[data-indicator="repulsion_btn"]');
        overlay = document.getElementById('repulsion-popover-overlay');
        textarea = document.getElementById('repulsion-reason');
        sendBtn = document.getElementById('repulsion-send');
        closeBtn = document.getElementById('repulsion-close');
        errorSlot = document.getElementById('repulsion-error');
        snippetSlot = document.getElementById('repulsion-snippets');
        responseContent = document.getElementById('response_content');

        if (!btn || !overlay || !textarea || !sendBtn) {
            console.warn('[repulsion-feedback] missing DOM nodes; feature disabled');
            return;
        }

        btn.style.display = 'none';

        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            openPopover();
        });

        sendBtn.addEventListener('click', (e) => {
            e.preventDefault();
            submitFeedback();
        });

        if (closeBtn) {
            closeBtn.addEventListener('click', (e) => {
                e.preventDefault();
                closePopover();
            });
        }

        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) closePopover();
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && overlay.classList.contains('active')) {
                closePopover();
            }
        });

        textarea.addEventListener('keydown', (e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                e.preventDefault();
                submitFeedback();
            }
        });

        wrapSendMessage();
        wrapCompleteStreamingResponse();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
