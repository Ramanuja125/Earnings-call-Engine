# """
# Phase 10 Runner - True Agentic AI
# Executes a self-learning agent that iteratively improves
# """

# from src.agentic.agentic_optimizer import AgenticAI

# def run_phase_10_agentic():
#     """
#     Execute Phase 10: True Agentic AI
#     """
#     print("\n" + "="*60)
#     print("PHASE 10: TRUE AGENTIC AI SYSTEM")
#     print("="*60)
#     print()
#     print("🤖 This is a TRULY AGENTIC system with:")
#     print()
#     print("   ✅ Autonomous Decision Making")
#     print("      Agent chooses which strategy to try")
#     print()
#     print("   ✅ Learning from Experience")
#     print("      Agent remembers what works and what doesn't")
#     print()
#     print("   ✅ Adaptive Strategy Selection")
#     print("      Agent shifts from exploration to exploitation")
#     print()
#     print("   ✅ Multi-Iteration Improvement")
#     print("      Agent iterates to find optimal approach")
#     print()
#     print("   ✅ Long-Term Memory")
#     print("      Agent saves learnings for future runs")
#     print()
#     print("   ✅ Gemini LLM Reasoning")  # ← new
#     print("      Agent uses AI to choose strategies intelligently")
#     print()
#     print("🔄 AGENTIC LOOP:")
#     print("   For each iteration:")
#     print("     1. Agent DECIDES which strategy to try (via Gemini)")
#     print("     2. Agent EXECUTES the strategy")
#     print("     3. Agent EVALUATES the outcome")
#     print("     4. Agent LEARNS from the result")
#     print("     5. Agent ADAPTS future behavior")
#     print()
#     print("⏱️  Estimated time: 15-20 minutes (10 iterations)")
#     print()

#     # ← THIS IS THE FIX: pass the API key here
#     agent = AgenticAI(api_key="AIzaSyBc3vKGuZJlJOJhMrw_SgdP4isCenJD6NY")
#     success = agent.run(max_iterations=10)

#     if success:
#         print("="*60)
#         print("🎉 AGENTIC AI COMPLETE!")
#         print("="*60)
#         print()
#         print("✅ Deliverables:")
#         print("   1. agent_memory.json - Agent's learned knowledge")
#         print("   2. agentic_ai_report.json - Full learning trajectory")
#         print()
#         print("📁 Location: results/agentic_ai/")
#         print()
#         print("🧠 The agent has autonomously discovered the optimal")
#         print("   strategy through iterative learning!")
#         print()

#     return success


# if __name__ == "__main__":
#     run_phase_10_agentic()

"""
Phase 10 Runner - True Agentic AI
Executes a self-learning agent that iteratively improves
"""

from src.agentic.agentic_optimizer import AgenticAI

def run_phase_10_agentic():
    """
    Execute Phase 10: True Agentic AI
    """
    print("\n" + "="*60)
    print("PHASE 10: TRUE AGENTIC AI SYSTEM")
    print("="*60)
    print()
    print("🤖 This is a TRULY AGENTIC system with:")
    print()
    print("   ✅ Autonomous Decision Making")
    print("      Agent chooses which strategy to try")
    print()
    print("   ✅ Learning from Experience")
    print("      Agent remembers what works and what doesn't")
    print()
    print("   ✅ Adaptive Strategy Selection")
    print("      Agent shifts from exploration to exploitation")
    print()
    print("   ✅ Multi-Iteration Improvement")
    print("      Agent iterates to find optimal approach")
    print()
    print("   ✅ Long-Term Memory")
    print("      Agent saves learnings for future runs")
    print()
    print("🔄 AGENTIC LOOP:")
    print("   For each iteration:")
    print("     1. Agent DECIDES which strategy to try")
    print("     2. Agent EXECUTES the strategy")
    print("     3. Agent EVALUATES the outcome")
    print("     4. Agent LEARNS from the result")
    print("     5. Agent ADAPTS future behavior")
    print()
    print("⏱️  Estimated time: 15-20 minutes (10 iterations)")
    print()
    
    agent = AgenticAI()
    success = agent.run(max_iterations=10)
    
    if success:
        print("="*60)
        print("🎉 AGENTIC AI COMPLETE!")
        print("="*60)
        print()
        print("✅ Deliverables:")
        print("   1. agent_memory.json - Agent's learned knowledge")
        print("   2. agentic_ai_report.json - Full learning trajectory")
        print()
        print("📁 Location: results/agentic_ai/")
        print()
        print("🧠 The agent has autonomously discovered the optimal")
        print("   strategy through iterative learning!")
        print()
    
    return success


if __name__ == "__main__":
    run_phase_10_agentic()