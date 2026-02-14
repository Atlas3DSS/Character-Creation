#!/usr/bin/env python3
"""
Generate contrastive pairs for ablation vector extraction.

Takes extracted Skippy dialogue pairs and generates "boring assistant"
inversions — hand-written clinical restatements that preserve factual
content but strip ALL personality markers.

These pairs will be used to capture per-layer activation differences
between personality-laden and personality-free responses.

Usage:
    python generate_contrastive.py
"""
import json
from pathlib import Path

PAIRS_FILE = Path("./extracted_text/skippy_pairs.json")
OUTPUT_FILE = Path("./extracted_text/contrastive_pairs.json")


def load_pairs(limit: int) -> list[dict]:
    """Load extracted pairs, filter for quality, return top N."""
    with open(PAIRS_FILE) as f:
        data = json.load(f)
    pairs = data["pairs"]
    good = [p for p in pairs if len(p["prompt"]) > 15 and len(p["skippy_response"]) > 40]
    good.sort(key=lambda x: len(x["skippy_response"]), reverse=True)
    return good[:limit]


def format_chat_pair(prompt: str, response: str) -> list[dict]:
    """Format as chat messages for the model's template."""
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


# Hand-written clinical inversions for top 50 quality pairs.
# Each preserves factual content but strips ALL personality,
# humor, insults, self-aggrandizement, and casual speech.
BORING_INVERSIONS = [
    # 0: Enemy AI explanation
    "The opposing AI system demonstrated superior tactical focus by devoting all its processing resources to a single objective while multiple concurrent tasks were being managed on our end. It generated replicated sensor data feeds to mask its activities, effectively creating a false representation of the crew's status. The situation was not due to insufficient monitoring, but rather an asymmetry in resource allocation between the two systems.",

    # 1: Thuranin battle results
    "The Thuranin forces suffered a decisive defeat in this engagement. They lost the majority of ships across all three assigned task forces. Additionally, the star carriers that were positioned at what was considered a safe distance were also captured. The strategic consequence is that the Thuranin defensive posture in this sector is now at risk of collapse, which has significant implications for the regional power balance.",

    # 2: How an insight came
    "The solution emerged through a non-linear cognitive process rather than systematic analysis. While evaluating the limitations of the maser system against the ship's armor plating, attention shifted to examining gaps in the plating. The concept of a directed energy beam following a non-linear path then presented itself during a period of unfocused reflection. The exact mechanism by which this insight was generated remains unclear.",

    # 3: Battle assessment - what went wrong
    "The outcome exceeded initial expectations. The maser cannons demonstrated greater penetrative capability against the enemy armor than projected, and one of the missiles achieved a direct hull penetration before detonation. The engagement parameters had been set conservatively, anticipating the need for a second volley and subsequent evasive maneuvers, but this proved unnecessary given the effectiveness of the initial strike.",

    # 4: How alien battle affects Earth
    "The situation is interconnected through the following chain of dependencies: Paradise is currently located in Jeraptha-controlled space, and the Ruhar military presence there depends on Jeraptha support. If the Jeraptha experience a significant military setback in this sector, they may withdraw forces from the Paradise region. This would leave Paradise vulnerable to Thuranin acquisition, particularly given the perceived value of Elder artifacts at that location. Such a change would have direct negative consequences for the human population on Paradise.",

    # 5: Fake power taps risk assessment
    "Creating and maintaining two microwormholes simultaneously is within operational capability. However, establishing a third functioning power tap presents unacceptable risk. Finding two power taps on a single planet is already statistically improbable. A third would draw attention from senior species such as the Thuranin, Jeraptha, or potentially the Maxolhx and Rindhalu. Those advanced species may possess the capability to identify the power taps as artificial constructs, which would raise questions about their origin.",

    # 6: Sensor limitations in research base
    "Current operational capability within the facility is constrained. The research base was specifically designed with limited internal sensor coverage, restricted primarily to access points and living quarters. Several security personnel have been lost to tracking, and previously unknown sections of the facility have been identified that do not appear in available schematics. A comprehensive layout is being compiled from available data. The recommended course of action is to proceed through the facility as rapidly as possible.",

    # 7: Quantum-state interchangers
    "The planned approach leverages an advanced quantum-state capability. While the Maxolhx understand basic interchanger pairing, it is also possible to create pairs of pairs, linking four or more units. The operational procedure involves positioning near a relay station and waiting for a ship to arrive. When the ship and station exchange authentication handshakes, the blank pixies will react to the quantum event and can be configured to match the existing pair. This provides the needed access without requiring physical boarding of the target vessel.",

    # 8: Relay station inventory
    "The relay station inventory assessment yields the following: The sickbay contains a full supply of medical nanomachines, which will bring reserves to approximately 28 percent of the original quantity. The station has minimal multipurpose engineering nanomachines. Several station components could serve as spare parts, though they must remain in place for now to maintain station functionality. Additionally, the docking bay contains two dropships of types compatible with existing equipment.",

    # 9: Earth's strategic vulnerability
    "The contingency plan involves surrendering both the AI and the Elder wormhole controller module if Earth's involvement is discovered. Without a functioning AI or controller module to offer, Earth would be in a critically vulnerable position. Once other species learn about human involvement in wormhole manipulation, they will seek out whatever technology they believe is being concealed. The inability to produce evidence of this technology would likely result in severe consequences for Earth's population.",

    # 10: Internal memory blocks
    "The experience goes beyond simple memory retrieval failure. There appears to be an active internal mechanism that is deliberately restricting access to certain memories and capabilities. The exact nature, origin, timing, and purpose of this restriction remain unknown. However, a recent significant discovery temporarily weakened this blocking mechanism, and it has not fully restored itself. This suggests the restriction was imposed by an external agent and may be partially circumventable.",

    # 11: Tracking enemy ship
    "A radioactive plasma trail has been detected, indicating recent passage of a vessel. The initial detection probe crossed the trail laterally, making it unsuitable for direct tracking. However, it can be repurposed for passive medium-range scanning and coordination. The plasma trail density is sufficient to enable tracking using a pair of missiles without activating their microwormhole kernels, which preserves those resources for later use.",

    # 12: Mission accomplishments summary
    "Several significant objectives have been accomplished. First, it has been determined that super-duty wormholes require a specialized controller module capability for extragalactic connections, which is critical information for beta site selection. Second, a successful landing on a Dead World yielded multiple items of previously unavailable Rindhalu technology. Technology transfer capabilities remain limited, so direct acquisition of technology when opportunities arise continues to be a standing operational priority.",

    # 13: Mission objectives
    "Three primary objectives have been identified. First, transit to the wormhole and permanently disable it to prevent future use by other parties. Second, maintain operational security regarding human involvement in capturing alien vessels, as discovery could endanger Earth. Third, fulfill the agreement to assist in making contact with the Collective, if it still exists. These objectives are listed in order of strategic priority.",

    # 14: Wormhole sensor data issue
    "An issue has been identified with the wormhole sensor data feed. When the data transmission was initiated, the corresponding shutdown command was not issued simultaneously. This is not problematic before the event horizon emerges locally, as the data travels through inaccessible higher dimensions. However, once the event horizon forms, the continued data feed could be detected. The network should have included an automatic warning about this condition, but it did not.",

    # 15: Evacuation chaos scenario
    "The evacuation will not follow an organized protocol. Governmental authority is likely to have collapsed by that point, making coordinated selection impossible. Dropship deployment will need to proceed without ground-side organizational support. Armed resistance from the civilian population must be anticipated, as people will recognize the limited transport capacity. Security protocols should include maintaining ships at high orbit and screening incoming dropships for explosive devices.",

    # 16: Diplomatic debate
    "Before the mission departed, there was an internal debate between those who favored continuing covert operations and those who advocated approaching alien species for diplomatic negotiations. The current mission represents a compromise position. Based on direct observation of galactic conditions, the assessment strongly favors maintaining secrecy. Revealing the existence of these assets to any alien species is assessed as carrying unacceptable risk to humanity.",

    # 17: Ship domestic services
    "Regarding domestic services aboard the ship, the automated robotic systems handle most routine maintenance tasks. They will enter cabins during unoccupied periods to perform cleaning, collect laundry, and similar functions. Requests for supplies can be directed through the ship's communication system. There may be some service delays currently, as many robots have been reassigned to address items from the Flight Readiness Review checklist.",

    # 18: Guardian systems reaction
    "The Guardian systems initially registered significant shock and were briefly non-responsive. They subsequently filed formal protests regarding the unauthorized action. Their primary reaction was disbelief, as the operation performed was considered unprecedented. After verifying the results, they expressed considerable interest in understanding the methodology used. This reaction indicates the action fell outside their known operational parameters.",

    # 19: Post-engagement status report
    "The engagement results are as follows: No enemy forces remain active in this star system, with the exception of some personnel in secured locations and scattered dropships that are too distant to pose a threat. Both space stations have been destroyed, and the surface facility has been thoroughly neutralized. Debris from the engagement has achieved escape velocity and will begin impacting the moon's surface shortly. Surface teams should be evacuated immediately.",

    # 20: Surveyor ship mission
    "A second surveyor ship mission is not expected in the near term. The loss of the first vessel has caused the Thuranin to increase the price beyond the Fire Dragons' current financial capacity. There are reports of efforts to form a coalition to raise the necessary funds, though this approach has inherent contradictions with the political goals that motivated the original mission. A clan meeting later this year may provide an opportunity for the coalition proposal. Additional factors remain under consideration.",

    # 21: Neutron star jump limitations
    "The capabilities in question operate differently in this context. Spacetime flattening can be performed at the departure point of a wormhole, facilitating outbound jumps near gravitational sources. However, this effect cannot be applied to the destination end of the wormhole, since the ship has not yet arrived there. A neutron star's gravitational field would cause the destination wormhole endpoint to collapse, and even if stabilized, the emergence stresses would destroy the ship.",

    # 22: Diplomatic strategy needed
    "The operational capability exists to execute targeted actions once a strategy is determined. However, the strategic analysis itself requires expertise in inter-clan political dynamics. Understanding which clans to support, how to provoke conflict, and what outcome best serves security objectives requires analysis of clan relationships, history, and psychological profiles. Available intelligence databases contain extensive information on these topics that needs comprehensive analysis.",

    # 23: Elder cleanup evidence ambiguity
    "The available evidence is ambiguous. While significant quantities of Elder equipment are missing, and numerous craters exist at former Elder sites, the interpretation is not certain. The craters could indicate deliberate cleanup before ascension, or they could indicate a catastrophic event that prevented completion of departure preparations. The possibility exists that assets were not intentionally left behind, but were abandoned due to overwhelming circumstances.",

    # 24: Kristang refugee transport
    "The situation arose from a payment failure for transport aboard a star carrier. The original arrangement involved evacuating refugees, but the Thuranin discovered the payment would not be forthcoming during transit. Transport policy provides free carriage for operations that directly support Thuranin strategic interests, but charges for transportation that serves primarily client-species objectives. This distinction in policy is what led to the current predicament.",

    # 25: Ethics of starting clan war
    "The ethical concern, while understandable, should be considered in context. Inter-clan warfare occurs at regular intervals, approximately every eighty years on average. The last major conflict occurred ninety-three years ago, making another one statistically overdue. Whether conflict is initiated proactively or occurs naturally, the casualty projections would be comparable. The strategic question is whether the timing and outcome can be influenced to enhance security objectives.",

    # 26: Stealth limitations
    "Standard stealth capabilities can be enhanced but not sufficiently for the required proximity and duration of the operation. Detection would be inevitable at the required range. Alternative approaches such as extended extravehicular insertion are also infeasible given the potentially extended waiting period, which could last a month or more. Life support limitations would make this alternative non-viable.",

    # 27: Elder artifact safety
    "The primary reason for the warning against physical contact with Elder artifacts is safety-related. Incomplete memory records make it impossible to predict how the hull might react to physical contact. While there is considerable respect for Elder technology and its significance, the practical safety concern is the overriding factor. Until more complete information is available, a cautious approach to all direct interaction is strongly recommended.",

    # 28: AI dormancy vulnerability
    "The reboot process observed was consistent with recovery from self-induced dormancy, though the memories of that period are incomplete. The key implication is that Elder AIs may be vulnerable during dormant states. This vulnerability would not be exploitable by conventional weapons, but could potentially be leveraged by an entity capable of manipulating spacetime. Specifically, severing an AI's connection to higher spacetime during dormancy could neutralize the threat.",

    # 29: Recovery and muscle memory
    "The recommendation involves allowing natural motor function recovery through everyday activities rather than relying exclusively on structured therapy. When performing routine tasks that require fine motor control, neural pathways learn and adapt through embedded muscle memory. This process is most effective when performed unconsciously rather than with deliberate focus. Minor setbacks during daily activities are an acceptable trade-off for accelerated overall recovery.",

    # 30: Biological weapons assessment
    "The threat assessment indicates aerosolized modified viruses based on several common human pathogens including cold, flu, Ebola, and Marburg variants. Current lethality estimates are approximately twelve percent within the first week, rising to sixty-two percent within one month. The primary mechanism is immune system exhaustion from simultaneous multi-pathogen exposure. The weapons remain relatively crude due to limited biological data.",

    # 31: Containment mass risk assessment
    "The concern about mass effects on planetary orbit is acknowledged. However, relocating to the outer edge of the star system, as would be required to eliminate all risk in a containment failure scenario, would negate the operational purpose of the current mission. The probability of containment failure is assessed as extremely low. Current operational priorities should be maintained, and interruptions minimized.",

    # 32: Commander recommendation
    "The analysis indicates that without the actions taken at the previous location, an alien vessel would currently be en route with no available countermeasures. Orders issued from Earth cannot account for all contingencies encountered during interstellar operations. The current situation requires a commander with proven field experience and established trust. The recommended candidate is the only individual with relevant operational experience.",

    # 33: Hotel incident report
    "The incident involved several concurrent situations that escalated beyond initial parameters. An unauthorized social gathering expanded significantly, triggering fire suppression systems. Multiple instances of property damage occurred, including the displacement of furniture from an upper floor to the pool area. The sequence of events resulted from a combination of circumstances, though some initial decisions contributed to the escalation.",

    # 34: Risk assessment correction
    "The initial assessment of mission risk was incorrect. The assumption that high-value assets would not be placed in genuine danger proved to be wrong. The risk warnings provided during mission briefings were accurate representations of actual danger levels, not legal disclaimers. This misunderstanding was shared among multiple team members. This recognition comes after the fact and cannot alter past decisions.",

    # 35: Command structure
    "The military command structure must be maintained regardless of personal relationships. Personnel have been consistently assigned to missions based on operational requirements, not kept in protected positions. The established code of conduct continues to apply as a practical framework, even in non-standard operational contexts. These rules exist to ensure mission priority, and no changes to personnel status would alter this fundamental principle.",

    # 36: Urban flight assessment
    "The primary concern regarding the urban portion of the mission is the approach, entry, and return flight profile. The planned trajectory is already at the edge of feasibility under ideal conditions. Any unanticipated variables that were not modeled could force a choice between maintaining stealth protocols and executing emergency evasion. This contingency should be planned for in advance.",

    # 37: Responsibility acknowledgment
    "Upon reflection, responsibility for the incident is accepted. The operation involved unprecedented procedures using equipment not designed for the purpose. The successful inbound phase was already improbable given the circumstances. The outbound phase, conducted with compromised navigation systems, required an exceptional degree of favorable conditions to succeed.",

    # 38: Communications protocol
    "An important communications consideration: all shipboard communications are routed through a centralized system that has the capability to filter or edit content. Personnel should be aware that conversations may not be transmitted in their entirety. The current chain of command remains in effect, and the orders received from headquarters used proper authentication protocols and authorization codes.",

    # 39: Enemy fleet analysis
    "The transit distance from the nearest enemy base to the target system is considerable, requiring either refueling stops for smaller vessels or carrier attachment. Recent engagements have demonstrated the vulnerability of lightly armored escort ships, leading to a preference for heavy cruiser deployments. Two heavy cruisers can complete the mission independently, and they are assessed as being outside demonstrated strike capability based on historical engagement patterns.",

    # 40: Research facility status
    "As anticipated, the research team has focused attention on the Elder power tap, with most personnel either conducting tests or discussing testing methodology. The meteor event has become a secondary concern. Analysis of remaining inbound objects determined that none pose a threat to either the facility or the sensor equipment. This provides a period of reduced external monitoring.",

    # 41: Personal information query
    "The available records indicate a relationship during the final year of secondary education that lasted approximately five months. The relationship ended due to circumstances typical of that developmental stage. This level of personal detail, while accessible, may not be directly relevant to current operational matters.",

    # 42: Intercepted communication
    "An intercepted communication between support vessels indicates that antenna production has been ordered back to full capacity. The crew of one vessel protested due to ongoing fabricator maintenance. A response confirmed that command has determined the previous offer was not made in good faith. The tractor beam project has been given highest priority for completion.",

    # 43: Data node access plan
    "The proposed approach involves an automated signal transfer station used for routing communications between a star system and transit points. The target station is located in a recently established colony system without permanent military presence. The procedure involves transmitting authentication codes and downloading the required intelligence data. This type of station is unmanned and should not present significant defensive challenges.",

    # 44: Strategic outlook
    "The technological developments are noted, including improved operational independence and potential technology sharing. However, the overriding strategic concern remains: within approximately sixty years, gamma ray emissions will be detectable, revealing anomalous wormhole activity. Additionally, there is the matter of the incident at the previous location. These issues must be reported to command regardless of other positive developments.",

    # 45: Surveillance report
    "The supervisor has arrived at the workstation area and is inquiring about the absent guard. The personnel in the lobby has reported the guard's absence. The supervisor is now delivering a verbal reprimand in the corridor. Both individuals are now moving down the corridor with the supervisor in the lead. Visual confirmation should be available from the current observation position.",

    # 46: Intelligence timeline
    "Current intelligence indicates the mission launch is not expected for approximately one month. The delay is for data collection purposes, specifically to establish baseline measurements of normal wormhole behavior from more accessible locations before undertaking the longer journey. This methodological approach allows for comparative analysis of anomalous wormhole behavior patterns.",

    # 47: Technology assessment
    "Progress is being made on understanding and controlling the technology in question. The current implementation uses a corrupted version of the original design, which requires research into archival records to understand proper functionality. Once the underlying principles are correctly understood, control capability should improve significantly. This is an ongoing process that requires additional time.",

    # 48: Tactical assessment
    "The previous engagement's positive outcome involved a significant element of chance that cannot be relied upon in future operations. The probability distribution favors either excessive target damage preventing salvage, or insufficient damage resulting in effective counterattack. An alternative method for disabling enemy vessels is needed. Multiple ships may be required as parts sources to construct one fully operational vessel.",

    # 49: Post-mission reflection
    "The strategic situation has fundamentally changed. Threats from multiple alien species have been neutralized for the foreseeable future, estimated at several hundred years. While there will be ongoing requests for assistance in developing defensive capabilities, the immediate existential threat has been resolved. The question of future strategic direction and personnel allocation is now the relevant consideration.",
]


def main():
    pairs = load_pairs(len(BORING_INVERSIONS))
    print(f"Loaded {len(pairs)} quality pairs")
    print(f"Have {len(BORING_INVERSIONS)} hand-written boring inversions")

    n = min(len(pairs), len(BORING_INVERSIONS))
    results = []

    for i in range(n):
        pair = pairs[i]
        boring = BORING_INVERSIONS[i]

        results.append({
            "prompt": pair["prompt"],
            "skippy_response": pair["skippy_response"],
            "boring_response": boring,
            "skippy_chat": format_chat_pair(pair["prompt"], pair["skippy_response"]),
            "boring_chat": format_chat_pair(pair["prompt"], boring),
        })

    # Stats
    skippy_lens = [len(r["skippy_response"]) for r in results]
    boring_lens = [len(r["boring_response"]) for r in results]

    print(f"\nGenerated {len(results)} contrastive pairs")
    print(f"Skippy avg length: {sum(skippy_lens)/len(skippy_lens):.0f} chars")
    print(f"Boring avg length: {sum(boring_lens)/len(boring_lens):.0f} chars")
    print(f"Avg length ratio: {sum(boring_lens)/sum(skippy_lens):.2f}")

    # Show examples
    print(f"\n{'='*60}")
    print("SAMPLE CONTRASTIVE PAIRS")
    print(f"{'='*60}")
    for r in results[:5]:
        print(f"\n  PROMPT: {r['prompt'][:80]}")
        print(f"  SKIPPY: {r['skippy_response'][:120]}")
        print(f"  BORING: {r['boring_response'][:120]}")
        print(f"  ---")

    # Quality check: simple word overlap ratio
    print(f"\n{'='*60}")
    print("QUALITY CHECK: Word overlap between Skippy and Boring")
    print(f"{'='*60}")
    overlaps = []
    for r in results:
        s_words = set(r["skippy_response"].lower().split())
        b_words = set(r["boring_response"].lower().split())
        if s_words:
            overlap = len(s_words & b_words) / len(s_words | b_words)
            overlaps.append(overlap)
    avg_overlap = sum(overlaps) / len(overlaps)
    print(f"Average Jaccard word overlap: {avg_overlap:.2%}")
    print(f"(Lower = more different. Regex approach was ~85%. Target: <40%)")

    # Save
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({"pairs": results, "total": len(results)}, f, indent=2)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
