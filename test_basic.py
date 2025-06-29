#!/usr/bin/env python3
"""
Basic functionality test for Network Traffic Manager
"""
import sys
import traceback
import numpy as np

def test_imports():
    """Test all critical imports"""
    print("Testing imports...")
    
    try:
        from config.config import Config
        print("✅ Config import OK")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from environments.network_env import NetworkEnvironment
        print("✅ NetworkEnvironment import OK")
    except Exception as e:
        print(f"❌ NetworkEnvironment import failed: {e}")
        return False
    
    try:
        from environments.topology_generator import TopologyGenerator
        print("✅ TopologyGenerator import OK")
    except Exception as e:
        print(f"❌ TopologyGenerator import failed: {e}")
        return False
    
    try:
        from environments.traffic_generator import TrafficGenerator
        print("✅ TrafficGenerator import OK")
    except Exception as e:
        print(f"❌ TrafficGenerator import failed: {e}")
        return False
    
    return True

def test_environment_basic():
    """Test basic environment functionality"""
    print("\nTesting environment creation...")
    
    try:
        from config.config import Config
        from environments.network_env import NetworkEnvironment
        
        config = Config()
        config.network.num_nodes = 6
        config.environment.max_episode_steps = 50
        
        env = NetworkEnvironment(config)
        print("✅ Environment created successfully")
        
        print(f"   Network: {env.network.number_of_nodes()} nodes, {env.network.number_of_edges()} edges")
        print(f"   Observation space: {env.observation_space.shape}")
        print(f"   Action space: {env.action_space}")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        traceback.print_exc()
        return False

def test_environment_step():
    """Test environment step functionality"""
    print("\nTesting environment steps...")
    
    try:
        from config.config import Config
        from environments.network_env import NetworkEnvironment
        
        config = Config()
        config.network.num_nodes = 6
        config.environment.max_episode_steps = 10
        
        env = NetworkEnvironment(config)
        
        # Reset environment
        obs, info = env.reset()
        print("✅ Environment reset OK")
        print(f"   Initial observation shape: {obs.shape}")
        print(f"   Info keys: {list(info.keys())}")
        
        # Take a few steps
        total_reward = 0
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"   Step {step+1}: reward={reward:.4f}, done={done}")
            
            if done or truncated:
                break
        
        print(f"✅ Environment steps completed. Total reward: {total_reward:.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Environment steps failed: {e}")
        traceback.print_exc()
        return False

def test_agent_creation():
    """Test agent creation"""
    print("\nTesting agent creation...")
    
    try:
        from config.config import Config
        from environments.network_env import NetworkEnvironment
        from agents.ppo_agent import PPOAgent
        
        config = Config()
        config.network.num_nodes = 6
        
        env = NetworkEnvironment(config)
        agent = PPOAgent(config, env.observation_space, env.action_space)
        
        print("✅ PPO Agent created successfully")
        
        # Test action selection
        obs, _ = env.reset()
        action, log_prob, value = agent.select_action(obs)
        
        print(f"   Action shape: {action.shape}")
        print(f"   Log prob: {log_prob:.4f}")
        print(f"   Value: {value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        traceback.print_exc()
        return False

def test_baseline_algorithms():
    """Test baseline algorithms"""
    print("\nTesting baseline algorithms...")
    
    try:
        from config.config import Config
        from environments.network_env import NetworkEnvironment
        from baselines.ospf_routing import OSPFRouter, ECMPRouter
        
        config = Config()
        config.network.num_nodes = 6
        
        env = NetworkEnvironment(config)
        
        # Test OSPF
        ospf = OSPFRouter(config)
        ospf.initialize(env.network)
        
        obs, _ = env.reset()
        action = ospf.get_action(obs)
        
        print("✅ OSPF router working")
        print(f"   Action shape: {action.shape}")
        
        # Test ECMP
        ecmp = ECMPRouter(config)
        ecmp.initialize(env.network)
        
        action = ecmp.get_action(obs)
        print("✅ ECMP router working")
        
        return True
        
    except Exception as e:
        print(f"❌ Baseline algorithms failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 NETWORK TRAFFIC MANAGER - BASIC FUNCTIONALITY TEST")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_environment_basic,
        test_environment_step,
        test_agent_creation,
        test_baseline_algorithms
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print("❌ Test failed, stopping here for debugging")
                break
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            traceback.print_exc()
            break
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The system is working correctly.")
        print("\nYou can now run:")
        print("  python main.py demo")
        print("  python main.py train --algorithm PPO --topology mesh --nodes 6")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())