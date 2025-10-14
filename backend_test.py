#!/usr/bin/env python3
"""
Backend API Testing Suite for Trading AI App
Tests all backend endpoints according to the review request.
"""

import requests
import json
import sys
import os
from typing import Dict, Any, Optional

# Configuration
BACKEND_URL = "https://tradewise-173.preview.emergentagent.com/api"
TIMEOUT = 30

class BackendTester:
    def __init__(self):
        self.session = requests.Session()
        self.session.timeout = TIMEOUT
        self.access_token: Optional[str] = None
        self.test_results = []
        
    def log_result(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        self.test_results.append({
            "test": test_name,
            "success": success,
            "details": details
        })
        print(f"{status}: {test_name}")
        if details:
            print(f"   Details: {details}")
    
    def test_server_health(self) -> bool:
        """Test 1: Server starts without SyntaxError and health endpoint works"""
        try:
            response = self.session.get(f"{BACKEND_URL}/")
            if response.status_code == 200:
                data = response.json()
                if "message" in data and "Trading AI backend is up" in data["message"]:
                    self.log_result("Server Health Check", True, f"Response: {data}")
                    return True
                else:
                    self.log_result("Server Health Check", False, f"Unexpected response: {data}")
                    return False
            else:
                self.log_result("Server Health Check", False, f"Status: {response.status_code}, Response: {response.text}")
                return False
        except Exception as e:
            self.log_result("Server Health Check", False, f"Exception: {str(e)}")
            return False
    
    def test_search_endpoint(self) -> bool:
        """Test 2: GET /api/search?q=RELI returns matching offline data"""
        try:
            response = self.session.get(f"{BACKEND_URL}/search", params={"q": "RELI"})
            if response.status_code == 200:
                data = response.json()
                if "results" in data and isinstance(data["results"], list):
                    # Check if RELIANCE is in results
                    found_reliance = any("RELI" in result.get("symbol", "").upper() or 
                                       "RELI" in result.get("name", "").upper() 
                                       for result in data["results"])
                    if found_reliance:
                        self.log_result("Search Endpoint", True, f"Found {len(data['results'])} results including RELIANCE")
                        return True
                    else:
                        self.log_result("Search Endpoint", False, f"No RELIANCE found in results: {data['results']}")
                        return False
                else:
                    self.log_result("Search Endpoint", False, f"Invalid response format: {data}")
                    return False
            else:
                self.log_result("Search Endpoint", False, f"Status: {response.status_code}, Response: {response.text}")
                return False
        except Exception as e:
            self.log_result("Search Endpoint", False, f"Exception: {str(e)}")
            return False
    
    def test_auth_signup(self) -> bool:
        """Test 3: POST /api/auth/signup with email/password returns access_token"""
        try:
            # Use a unique email for testing
            test_email = f"test_user_{os.urandom(4).hex()}@example.com"
            test_password = "TestPassword123!"
            
            payload = {
                "email": test_email,
                "password": test_password
            }
            
            response = self.session.post(f"{BACKEND_URL}/auth/signup", json=payload)
            if response.status_code == 200:
                data = response.json()
                if "access_token" in data and "token_type" in data:
                    self.access_token = data["access_token"]
                    self.log_result("Auth Signup", True, f"Token received, type: {data['token_type']}")
                    return True
                else:
                    self.log_result("Auth Signup", False, f"Missing token in response: {data}")
                    return False
            else:
                self.log_result("Auth Signup", False, f"Status: {response.status_code}, Response: {response.text}")
                return False
        except Exception as e:
            self.log_result("Auth Signup", False, f"Exception: {str(e)}")
            return False
    
    def test_auth_me(self) -> bool:
        """Test 4: GET /api/auth/me with bearer token works and includes default profile"""
        if not self.access_token:
            self.log_result("Auth Me", False, "No access token available")
            return False
        
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = self.session.get(f"{BACKEND_URL}/auth/me", headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ["id", "email", "profile"]
                if all(field in data for field in required_fields):
                    profile = data["profile"]
                    if isinstance(profile, dict) and "provider" in profile:
                        self.log_result("Auth Me", True, f"User data: {data}")
                        return True
                    else:
                        self.log_result("Auth Me", False, f"Invalid profile format: {profile}")
                        return False
                else:
                    self.log_result("Auth Me", False, f"Missing required fields: {data}")
                    return False
            else:
                self.log_result("Auth Me", False, f"Status: {response.status_code}, Response: {response.text}")
                return False
        except Exception as e:
            self.log_result("Auth Me", False, f"Exception: {str(e)}")
            return False
    
    def test_profile_get(self) -> bool:
        """Test 5: GET /api/profile with bearer token reads profile"""
        if not self.access_token:
            self.log_result("Profile GET", False, "No access token available")
            return False
        
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = self.session.get(f"{BACKEND_URL}/profile", headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and "provider" in data:
                    self.log_result("Profile GET", True, f"Profile data: {data}")
                    return True
                else:
                    self.log_result("Profile GET", False, f"Invalid profile format: {data}")
                    return False
            else:
                self.log_result("Profile GET", False, f"Status: {response.status_code}, Response: {response.text}")
                return False
        except Exception as e:
            self.log_result("Profile GET", False, f"Exception: {str(e)}")
            return False
    
    def test_profile_put(self) -> bool:
        """Test 6: PUT /api/profile with bearer token writes profile"""
        if not self.access_token:
            self.log_result("Profile PUT", False, "No access token available")
            return False
        
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            new_profile = {
                "provider": "openai",
                "model": "gpt-4"
            }
            
            response = self.session.put(f"{BACKEND_URL}/profile", headers=headers, json=new_profile)
            
            if response.status_code == 200:
                data = response.json()
                if data == new_profile:
                    self.log_result("Profile PUT", True, f"Profile updated: {data}")
                    return True
                else:
                    self.log_result("Profile PUT", False, f"Profile not updated correctly: {data}")
                    return False
            else:
                self.log_result("Profile PUT", False, f"Status: {response.status_code}, Response: {response.text}")
                return False
        except Exception as e:
            self.log_result("Profile PUT", False, f"Exception: {str(e)}")
            return False
    
    def test_telegram_config_post(self) -> bool:
        """Test 7: POST /api/alerts/telegram/config saves values"""
        if not self.access_token:
            self.log_result("Telegram Config POST", False, "No access token available")
            return False
        
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            config = {
                "chat_id": "123456789",
                "enabled": True,
                "buy_threshold": 85,
                "sell_threshold": 55,
                "frequency_min": 30,
                "quiet_start_hour": 23,
                "quiet_end_hour": 6,
                "timezone": "Asia/Kolkata"
            }
            
            response = self.session.post(f"{BACKEND_URL}/alerts/telegram/config", headers=headers, json=config)
            
            if response.status_code == 200:
                data = response.json()
                if "ok" in data and data["ok"]:
                    self.log_result("Telegram Config POST", True, f"Config saved: {data}")
                    return True
                else:
                    self.log_result("Telegram Config POST", False, f"Config not saved: {data}")
                    return False
            else:
                self.log_result("Telegram Config POST", False, f"Status: {response.status_code}, Response: {response.text}")
                return False
        except Exception as e:
            self.log_result("Telegram Config POST", False, f"Exception: {str(e)}")
            return False
    
    def test_telegram_config_get(self) -> bool:
        """Test 8: GET /api/alerts/telegram/config reflects saved values"""
        if not self.access_token:
            self.log_result("Telegram Config GET", False, "No access token available")
            return False
        
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = self.session.get(f"{BACKEND_URL}/alerts/telegram/config", headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                # Check if the previously saved values are reflected
                expected_values = {
                    "chat_id": "123456789",
                    "enabled": True,
                    "buy_threshold": 85,
                    "sell_threshold": 55
                }
                
                all_match = all(data.get(key) == value for key, value in expected_values.items())
                if all_match:
                    self.log_result("Telegram Config GET", True, f"Config retrieved: {data}")
                    return True
                else:
                    self.log_result("Telegram Config GET", False, f"Config mismatch. Expected: {expected_values}, Got: {data}")
                    return False
            else:
                self.log_result("Telegram Config GET", False, f"Status: {response.status_code}, Response: {response.text}")
                return False
        except Exception as e:
            self.log_result("Telegram Config GET", False, f"Exception: {str(e)}")
            return False
    
    def test_watchlist_put(self) -> bool:
        """Test 9: PUT /api/portfolio/watchlist persists symbols"""
        if not self.access_token:
            self.log_result("Watchlist PUT", False, "No access token available")
            return False
        
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            watchlist = {
                "symbols": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
            }
            
            response = self.session.put(f"{BACKEND_URL}/portfolio/watchlist", headers=headers, json=watchlist)
            
            if response.status_code == 200:
                data = response.json()
                if "ok" in data and data["ok"] and "symbols" in data:
                    self.log_result("Watchlist PUT", True, f"Watchlist updated: {data}")
                    return True
                else:
                    self.log_result("Watchlist PUT", False, f"Watchlist not updated: {data}")
                    return False
            else:
                self.log_result("Watchlist PUT", False, f"Status: {response.status_code}, Response: {response.text}")
                return False
        except Exception as e:
            self.log_result("Watchlist PUT", False, f"Exception: {str(e)}")
            return False
    
    def test_watchlist_get(self) -> bool:
        """Test 10: GET /api/portfolio/watchlist returns persisted symbols"""
        if not self.access_token:
            self.log_result("Watchlist GET", False, "No access token available")
            return False
        
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = self.session.get(f"{BACKEND_URL}/portfolio/watchlist", headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                if "symbols" in data and isinstance(data["symbols"], list):
                    expected_symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
                    if set(data["symbols"]) == set(expected_symbols):
                        self.log_result("Watchlist GET", True, f"Watchlist retrieved: {data}")
                        return True
                    else:
                        self.log_result("Watchlist GET", False, f"Watchlist mismatch. Expected: {expected_symbols}, Got: {data['symbols']}")
                        return False
                else:
                    self.log_result("Watchlist GET", False, f"Invalid watchlist format: {data}")
                    return False
            else:
                self.log_result("Watchlist GET", False, f"Status: {response.status_code}, Response: {response.text}")
                return False
        except Exception as e:
            self.log_result("Watchlist GET", False, f"Exception: {str(e)}")
            return False
    
    def test_analyze_endpoint(self) -> bool:
        """Test 11: POST /api/analyze responds with analysis structure"""
        try:
            payload = {
                "symbol": "RELIANCE.NS",
                "timeframe": "weekly",
                "market": "IN",
                "source": "yahoo"
            }
            
            response = self.session.post(f"{BACKEND_URL}/analyze", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ["symbol", "timeframe", "action", "confidence", "reasons", "indicators_snapshot"]
                if all(field in data for field in required_fields):
                    self.log_result("Analyze Endpoint", True, f"Analysis completed for {data['symbol']}: {data['action']} with {data['confidence']}% confidence")
                    return True
                else:
                    self.log_result("Analyze Endpoint", False, f"Missing required fields in response: {data}")
                    return False
            else:
                self.log_result("Analyze Endpoint", False, f"Status: {response.status_code}, Response: {response.text}")
                return False
        except Exception as e:
            self.log_result("Analyze Endpoint", False, f"Exception: {str(e)}")
            return False
    
    def test_signal_current(self) -> bool:
        """Test 12: GET /api/signal/current responds with signal structure"""
        try:
            params = {
                "symbol": "TCS.NS",
                "timeframe": "weekly",
                "source": "yahoo"
            }
            
            response = self.session.get(f"{BACKEND_URL}/signal/current", params=params)
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ["symbol", "timeframe", "action", "confidence", "reasons"]
                if all(field in data for field in required_fields):
                    self.log_result("Signal Current", True, f"Signal for {data['symbol']}: {data['action']} with {data['confidence']}% confidence")
                    return True
                else:
                    self.log_result("Signal Current", False, f"Missing required fields in response: {data}")
                    return False
            else:
                self.log_result("Signal Current", False, f"Status: {response.status_code}, Response: {response.text}")
                return False
        except Exception as e:
            self.log_result("Signal Current", False, f"Exception: {str(e)}")
            return False
    
    def test_strategy_suggest(self) -> bool:
        """Test 13: POST /api/strategy/suggest returns picks array and used_ai boolean"""
        try:
            payload = {
                "filters": {
                    "risk_tolerance": "medium",
                    "horizon": "weekly",
                    "asset_classes": ["stocks"],
                    "market": "IN",
                    "momentum_preference": True,
                    "value_preference": False,
                    "allocation": {"stocks": 100},
                    "caps_allocation": {"largecap": 60, "midcap": 25, "smallcap": 15}
                },
                "top_n": 5,
                "source": "yahoo"
            }
            
            response = self.session.post(f"{BACKEND_URL}/strategy/suggest", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if "picks" in data and "used_ai" in data:
                    picks = data["picks"]
                    if isinstance(picks, list) and isinstance(data["used_ai"], bool):
                        self.log_result("Strategy Suggest", True, f"Strategy returned {len(picks)} picks, AI used: {data['used_ai']}")
                        return True
                    else:
                        self.log_result("Strategy Suggest", False, f"Invalid picks or used_ai format: {data}")
                        return False
                else:
                    self.log_result("Strategy Suggest", False, f"Missing picks or used_ai in response: {data}")
                    return False
            else:
                self.log_result("Strategy Suggest", False, f"Status: {response.status_code}, Response: {response.text}")
                return False
        except Exception as e:
            self.log_result("Strategy Suggest", False, f"Exception: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all backend tests"""
        print("=" * 60)
        print("BACKEND API TESTING SUITE")
        print("=" * 60)
        
        tests = [
            self.test_server_health,
            self.test_search_endpoint,
            self.test_auth_signup,
            self.test_auth_me,
            self.test_profile_get,
            self.test_profile_put,
            self.test_telegram_config_post,
            self.test_telegram_config_get,
            self.test_watchlist_put,
            self.test_watchlist_get,
            self.test_analyze_endpoint,
            self.test_signal_current,
            self.test_strategy_suggest
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                if test():
                    passed += 1
            except Exception as e:
                print(f"❌ FAIL: {test.__name__} - Exception: {str(e)}")
        
        print("\n" + "=" * 60)
        print(f"SUMMARY: {passed}/{total} tests passed")
        print("=" * 60)
        
        # Print detailed results
        print("\nDETAILED RESULTS:")
        for result in self.test_results:
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"{status}: {result['test']}")
            if result["details"]:
                print(f"   {result['details']}")
        
        return passed == total

if __name__ == "__main__":
    tester = BackendTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)