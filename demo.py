"""
Interactive Demo for @AmazonHelp AI Customer Support Agent.
Run this script to test the agent interactively with preset or custom tweets.
Usage: python demo.py
"""

import sys
from src.agent import SupportAgent
from src.config import Intent

SAMPLE_TWEETS = [
    "Where is my package? It was supposed to arrive yesterday but tracking hasn't updated!",
    "The phone charger I ordered caught fire and scorched my bedroom wall. I am suing Amazon!",
    "Someone hacked my account, changed the email, and bought $800 in gift cards!! Help!!",
    "I returned the shoes 10 days ago via UPS and tracking says delivered, but where is my refund?",
    "Received my package today, opened the sealed box and it was filled with soap bars instead of iPhone!",
    "How do I update the expiration date on my credit card?",
    "Thanks Amazon for delivering my package to my neighbor's roof! Stellar service right there."
]

def run_demo():
    print("=" * 80)
    print("  @AmazonHelp AI CUSTOMER SUPPORT AGENT - INTERACTIVE DEMO")
    print("=" * 80)
    print("Loading AI Support Agent...")
    agent = SupportAgent()
    print("[OK] Agent ready!\n")

    while True:
        print("-" * 80)
        print("Choose an option:")
        print("  [1-7] Run a preset sample tweet")
        print("  [8]   Type your own custom customer tweet")
        print("  [q]   Quit")
        print("-" * 80)
        
        choice = input("Enter choice: ").strip()
        
        if choice.lower() in ["q", "quit", "exit"]:
            print("\nExiting demo. Goodbye!")
            break
            
        tweet_text = ""
        if choice in ["1", "2", "3", "4", "5", "6", "7"]:
            tweet_text = SAMPLE_TWEETS[int(choice) - 1]
            print(f"\n[Selected Sample {choice}]: \"{tweet_text}\"")
        elif choice == "8":
            tweet_text = input("\nEnter customer tweet: ").strip()
            if not tweet_text:
                print("Empty input. Try again.")
                continue
        else:
            print("Invalid selection. Please enter 1-8 or q.")
            continue

        print("\nProcessing inquiry...")
        result = agent.process_tweet(tweet_text)

        print("\n" + "=" * 80)
        print(f"Customer Tweet:       {result['customer_text']}")
        print(f"Predicted Intent:     {result['predicted_intent']} (Confidence: {result['intent_confidence']:.2%})")
        print(f"Escalation Decision:  {result['escalation_decision']}")
        print(f"Escalation Reason:    {result['escalation_reason']}")
        print(f"Risk Score:           {result['risk_score']:.2f}")
        print("-" * 80)
        print(f"Drafted Brand Reply:\n  \"{result['drafted_reply']}\"")
        print("=" * 80 + "\n")

if __name__ == "__main__":
    run_demo()
