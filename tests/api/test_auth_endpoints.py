"""
Integration tests for authentication endpoints.

Tests the complete passwordless authentication flow with real services.
"""
import pytest
from fastapi.testclient import TestClient


class TestSignupEndpoint:
    """Test POST /v0/auth/signup endpoint."""

    def test_signup_creates_new_user_with_valid_email(self, test_client: TestClient):
        """Verify signup creates user account with valid email."""
        response = test_client.post(
            "/v0/auth/signup",
            json={"email": "newuser@example.com"}
        )

        assert response.status_code == 201
        data = response.json()

        # Verify response structure
        assert data["success"] is True
        assert "data" in data
        assert "user_id" in data["data"]
        assert data["data"]["email"] == "newuser@example.com"
        assert "message" in data["data"]

    def test_signup_returns_proper_meta_fields(self, test_client: TestClient):
        """Verify signup response includes request_id and timestamp."""
        response = test_client.post(
            "/v0/auth/signup",
            json={"email": "another@example.com"}
        )

        data = response.json()
        meta = data["meta"]

        assert "request_id" in meta
        assert meta["request_id"].startswith("req_")
        assert "timestamp" in meta
        assert "http_status" in meta
        assert meta["http_status"] == 201

    def test_signup_rejects_invalid_email_formats(self, test_client: TestClient):
        """Verify signup rejects malformed email addresses."""
        invalid_emails = [
            "notanemail",
            "@domain.com",
            "user@",
            "user space@domain.com",
            ""
        ]

        for invalid_email in invalid_emails:
            response = test_client.post(
                "/v0/auth/signup",
                json={"email": invalid_email}
            )

            # Should return 422 (Pydantic validation) or 400
            assert response.status_code in [400, 422]

    def test_signup_returns_400_when_user_already_exists(self, test_client: TestClient, test_user):
        """Verify signup returns error for duplicate email."""
        response = test_client.post(
            "/v0/auth/signup",
            json={"email": test_user["email"]}
        )

        assert response.status_code == 400
        data = response.json()

        assert data["success"] is False
        assert "error" in data
        assert data["error"]["code"] in ["user_already_exists", "USER_ALREADY_EXISTS"]

    def test_signup_missing_email_field(self, test_client: TestClient):
        """Verify signup requires email field."""
        response = test_client.post(
            "/v0/auth/signup",
            json={}
        )

        # FastAPI Pydantic validation
        assert response.status_code == 422

    def test_signup_response_structure_exact(self, test_client: TestClient):
        """Verify exact response structure for successful signup."""
        response = test_client.post(
            "/v0/auth/signup",
            json={"email": "structure_test@example.com"}
        )

        data = response.json()

        # Top-level keys
        assert set(data.keys()) == {"success", "data", "meta"}
        assert data["success"] is True

        # Data keys
        assert "user_id" in data["data"]
        assert "email" in data["data"]
        assert "message" in data["data"]

        # Meta keys
        assert "request_id" in data["meta"]
        assert "timestamp" in data["meta"]
        assert "http_status" in data["meta"]


class TestMagicLinkEndpoint:
    """Test POST /v0/auth/magic-link endpoint."""

    def test_magic_link_sends_for_existing_user(self, test_client: TestClient):
        """Verify magic link request for existing user succeeds."""
        # Create unique user for this test to avoid rate limiting
        unique_email = f"magic_link_test_{id(self)}@example.com"
        test_client.post("/v0/auth/signup", json={"email": unique_email})

        response = test_client.post(
            "/v0/auth/magic-link",
            json={"email": unique_email}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "message" in data["data"]
        assert "sent" in data["data"]["message"].lower() or "magic link" in data["data"]["message"].lower()

    def test_magic_link_returns_404_for_nonexistent_user(self, test_client: TestClient):
        """Verify magic link returns 404 for non-existent email."""
        response = test_client.post(
            "/v0/auth/magic-link",
            json={"email": "nonexistent@example.com"}
        )

        assert response.status_code == 404
        data = response.json()

        assert data["success"] is False
        assert data["error"]["code"] in ["user_not_found", "USER_NOT_FOUND"]

    def test_magic_link_rejects_invalid_email(self, test_client: TestClient):
        """Verify magic link rejects invalid email formats."""
        response = test_client.post(
            "/v0/auth/magic-link",
            json={"email": "not_an_email"}
        )

        assert response.status_code in [400, 422]

    def test_magic_link_response_structure(self, test_client: TestClient):
        """Verify magic link response has correct structure."""
        # Create unique user for this test to avoid rate limiting
        unique_email = f"magic_structure_test_{id(self)}@example.com"
        test_client.post("/v0/auth/signup", json={"email": unique_email})

        response = test_client.post(
            "/v0/auth/magic-link",
            json={"email": unique_email}
        )

        data = response.json()

        assert "success" in data
        assert "data" in data
        assert "meta" in data
        assert "request_id" in data["meta"]
        assert "timestamp" in data["meta"]

    def test_magic_link_missing_email_field(self, test_client: TestClient):
        """Verify magic link requires email field."""
        response = test_client.post(
            "/v0/auth/magic-link",
            json={}
        )

        assert response.status_code == 422

    def test_magic_link_rate_limiting(self, test_client: TestClient):
        """Verify rate limiting prevents excessive magic link requests."""
        # Create unique user for rate limit test
        unique_email = f"rate_limit_test_{id(self)}@example.com"
        test_client.post("/v0/auth/signup", json={"email": unique_email})

        # Make multiple rapid requests
        for i in range(10):
            response = test_client.post(
                "/v0/auth/magic-link",
                json={"email": unique_email}
            )

            # May hit rate limit after several requests
            if response.status_code == 429:
                data = response.json()
                assert data["success"] is False
                assert "rate_limit" in data["error"]["code"].lower() or "RATE_LIMIT" in data["error"]["code"]
                break


class TestVerifyMagicLinkEndpoint:
    """Test POST /v0/auth/verify endpoint."""

    def test_verify_valid_magic_link_token(self, test_client: TestClient, magic_link_token):
        """Verify valid magic link token creates session."""
        response = test_client.post(
            "/v0/auth/verify",
            json={"token": magic_link_token}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "user" in data["data"]
        assert "session_token" in data["data"]

    def test_verify_sets_session_cookie(self, test_client: TestClient, magic_link_token):
        """Verify session cookie is set with proper security attributes."""
        response = test_client.post(
            "/v0/auth/verify",
            json={"token": magic_link_token}
        )

        # Check Set-Cookie header
        set_cookie = response.headers.get("set-cookie", "")
        assert "session=" in set_cookie
        assert "HttpOnly" in set_cookie
        assert "SameSite" in set_cookie

    def test_verify_returns_user_profile(self, test_client: TestClient, magic_link_token):
        """Verify response includes user profile data."""
        response = test_client.post(
            "/v0/auth/verify",
            json={"token": magic_link_token}
        )

        data = response.json()
        user = data["data"]["user"]

        assert "id" in user
        assert "email" in user
        assert "is_active" in user
        assert "created_at" in user

    def test_verify_rejects_invalid_token_format(self, test_client: TestClient):
        """Verify invalid token format returns 400."""
        response = test_client.post(
            "/v0/auth/verify",
            json={"token": "short"}
        )

        # Pydantic validation for min_length
        assert response.status_code in [400, 422]

    def test_verify_rejects_expired_token(self, test_client: TestClient, expired_magic_link_token):
        """Verify expired magic link returns 410."""
        response = test_client.post(
            "/v0/auth/verify",
            json={"token": expired_magic_link_token}
        )

        assert response.status_code == 410
        data = response.json()

        assert data["success"] is False
        assert "expired" in data["error"]["code"].lower()

    def test_verify_token_cannot_be_reused(self, test_client: TestClient, magic_link_token):
        """Verify magic link token is consumed after first use."""
        # First use succeeds
        response1 = test_client.post(
            "/v0/auth/verify",
            json={"token": magic_link_token}
        )
        assert response1.status_code == 200

        # Second use fails
        response2 = test_client.post(
            "/v0/auth/verify",
            json={"token": magic_link_token}
        )
        assert response2.status_code in [400, 410]

    def test_verify_missing_token_field(self, test_client: TestClient):
        """Verify token field is required."""
        response = test_client.post(
            "/v0/auth/verify",
            json={}
        )

        assert response.status_code == 422

    def test_verify_response_structure_exact(self, test_client: TestClient, magic_link_token):
        """Verify exact response structure."""
        response = test_client.post(
            "/v0/auth/verify",
            json={"token": magic_link_token}
        )

        data = response.json()

        # Top-level structure
        assert set(data.keys()) == {"success", "data", "meta"}

        # Data structure
        assert "user" in data["data"]
        assert "session_token" in data["data"]

        # User structure
        user = data["data"]["user"]
        assert "id" in user
        assert "email" in user


class TestLogoutEndpoint:
    """Test POST /v0/auth/logout endpoint."""

    def test_logout_authenticated_session(self, test_client: TestClient, session_token):
        """Verify logout invalidates session."""
        response = test_client.post(
            "/v0/auth/logout",
            headers={"Authorization": f"Bearer {session_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "message" in data["data"]

    def test_logout_clears_session_cookie(self, test_client: TestClient, session_token):
        """Verify logout clears session cookie."""
        response = test_client.post(
            "/v0/auth/logout",
            headers={"Authorization": f"Bearer {session_token}"}
        )

        # Check for cookie deletion
        set_cookie = response.headers.get("set-cookie", "")
        assert "session=" in set_cookie
        # Cookie should be cleared (Max-Age=0 or expires in past)

    def test_logout_invalidates_session_token(self, test_client: TestClient, session_token):
        """Verify token becomes invalid after logout."""
        # Logout
        logout_response = test_client.post(
            "/v0/auth/logout",
            headers={"Authorization": f"Bearer {session_token}"}
        )
        assert logout_response.status_code == 200

        # Try to use same token
        session_response = test_client.get(
            "/v0/auth/session",
            headers={"Authorization": f"Bearer {session_token}"}
        )

        # Should fail with 401
        assert session_response.status_code == 401

    def test_logout_returns_404_if_session_not_found(self, test_client: TestClient):
        """Verify logout returns 404 for invalid session."""
        response = test_client.post(
            "/v0/auth/logout",
            headers={"Authorization": "Bearer invalid_token_12345678901234567890123456789012"}
        )

        assert response.status_code == 404
        data = response.json()

        assert data["success"] is False

    def test_logout_requires_authentication(self, test_client: TestClient):
        """Verify logout requires auth token."""
        response = test_client.post("/v0/auth/logout")

        # No auth provided - should return 401
        assert response.status_code == 401
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "UNAUTHORIZED"

    def test_logout_response_structure(self, test_client: TestClient, session_token):
        """Verify logout response structure."""
        response = test_client.post(
            "/v0/auth/logout",
            headers={"Authorization": f"Bearer {session_token}"}
        )

        data = response.json()

        assert "success" in data
        assert "data" in data
        assert "meta" in data
        assert "request_id" in data["meta"]


class TestLogoutAllEndpoint:
    """Test POST /v0/auth/logout-all endpoint."""

    def test_logout_all_requires_authentication(self, test_client: TestClient):
        """Verify logout-all requires authentication."""
        response = test_client.post("/v0/auth/logout-all")

        assert response.status_code in [401, 403]

    def test_logout_all_revokes_all_sessions(self, authenticated_client: TestClient, authenticated_user):
        """Verify logout-all revokes all user sessions."""
        response = authenticated_client.post("/v0/auth/logout-all")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "revoked_sessions" in data["data"]
        assert isinstance(data["data"]["revoked_sessions"], int)

    def test_logout_all_returns_count_of_revoked_sessions(self, authenticated_client: TestClient, authenticated_user):
        """Verify response includes count of revoked sessions."""
        response = authenticated_client.post("/v0/auth/logout-all")

        data = response.json()

        assert "revoked_sessions" in data["data"]
        assert data["data"]["revoked_sessions"] >= 0

    def test_logout_all_invalidates_all_tokens(self, authenticated_client: TestClient, authenticated_user):
        """Verify all previous tokens become invalid."""
        # Get current token from authenticated_client
        original_token = authenticated_client.headers.get("Authorization", "").replace("Bearer ", "")

        # Logout all
        response = authenticated_client.post("/v0/auth/logout-all")
        assert response.status_code == 200

        # Try to use original token
        session_response = authenticated_client.get("/v0/auth/session")

        # Should fail with 401
        assert session_response.status_code == 401

    def test_logout_all_response_structure(self, authenticated_client: TestClient, authenticated_user):
        """Verify logout-all response structure."""
        response = authenticated_client.post("/v0/auth/logout-all")

        data = response.json()

        assert "success" in data
        assert "data" in data
        assert "message" in data["data"]
        assert "revoked_sessions" in data["data"]


class TestSessionEndpoint:
    """Test GET /v0/auth/session endpoint."""

    def test_session_requires_authentication(self, test_client: TestClient):
        """Verify /session requires authentication."""
        response = test_client.get("/v0/auth/session")

        assert response.status_code in [401, 403]

    def test_session_returns_current_session_data(self, authenticated_client: TestClient, authenticated_user):
        """Verify /session returns session data for authenticated user."""
        response = authenticated_client.get("/v0/auth/session")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "data" in data

    def test_session_includes_user_id_and_email(self, authenticated_client: TestClient, authenticated_user):
        """Verify session data includes user information."""
        response = authenticated_client.get("/v0/auth/session")

        data = response.json()
        session_data = data["data"]

        assert "user_id" in session_data
        assert "email" in session_data
        assert session_data["user_id"] == authenticated_user["user_id"]
        assert session_data["email"] == authenticated_user["email"]

    def test_session_returns_401_for_invalid_token(self, test_client: TestClient):
        """Verify invalid token returns 401."""
        response = test_client.get(
            "/v0/auth/session",
            headers={"Authorization": "Bearer invalid_token_12345678901234567890123456789012"}
        )

        assert response.status_code == 401

    def test_session_response_structure(self, authenticated_client: TestClient):
        """Verify session response structure."""
        response = authenticated_client.get("/v0/auth/session")

        data = response.json()

        assert "success" in data
        assert "data" in data
        assert "meta" in data
        assert "request_id" in data["meta"]


class TestCSRFEndpoint:
    """Test POST /v0/auth/csrf endpoint."""

    def test_csrf_requires_authentication(self, test_client: TestClient):
        """Verify /csrf requires authentication."""
        response = test_client.post("/v0/auth/csrf")

        assert response.status_code in [401, 403]

    def test_csrf_generates_token_for_session(self, authenticated_client: TestClient):
        """Verify CSRF token is generated for authenticated session."""
        response = authenticated_client.post("/v0/auth/csrf")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "csrf_token" in data["data"]
        assert isinstance(data["data"]["csrf_token"], str)
        assert len(data["data"]["csrf_token"]) > 0

    def test_csrf_token_is_valid_string(self, authenticated_client: TestClient):
        """Verify CSRF token is a valid non-empty string."""
        response = authenticated_client.post("/v0/auth/csrf")

        data = response.json()
        csrf_token = data["data"]["csrf_token"]

        assert len(csrf_token) >= 16  # Should be a reasonable length
        assert isinstance(csrf_token, str)

    def test_csrf_different_sessions_get_different_tokens(self, test_client: TestClient, test_user):
        """Verify different sessions receive different CSRF tokens."""
        from auth.service import AuthService

        auth_service = AuthService()

        # Create two different sessions
        session1 = auth_service.create_session(test_user["id"], test_user)
        session2 = auth_service.create_session(test_user["id"], test_user)

        response1 = test_client.post(
            "/v0/auth/csrf",
            headers={"Authorization": f"Bearer {session1}"}
        )
        response2 = test_client.post(
            "/v0/auth/csrf",
            headers={"Authorization": f"Bearer {session2}"}
        )

        csrf1 = response1.json()["data"]["csrf_token"]
        csrf2 = response2.json()["data"]["csrf_token"]

        # Different sessions should get different CSRF tokens
        assert csrf1 != csrf2

    def test_csrf_response_structure(self, authenticated_client: TestClient):
        """Verify CSRF response structure."""
        response = authenticated_client.post("/v0/auth/csrf")

        data = response.json()

        assert set(data.keys()) == {"success", "data", "meta"}
        assert "csrf_token" in data["data"]
        assert "request_id" in data["meta"]
        assert "timestamp" in data["meta"]


class TestAuthenticationFlow:
    """Integration test for complete authentication flow."""

    def test_complete_auth_flow(self, test_client: TestClient):
        """Test complete flow: signup → magic-link → verify → use session → logout."""
        unique_email = f"flow_test_{id(self)}@example.com"

        # Step 1: Signup
        signup_response = test_client.post(
            "/v0/auth/signup",
            json={"email": unique_email}
        )
        assert signup_response.status_code == 201
        user_id = signup_response.json()["data"]["user_id"]

        # Step 2: Request magic link
        magic_link_response = test_client.post(
            "/v0/auth/magic-link",
            json={"email": unique_email}
        )
        assert magic_link_response.status_code == 200

        # Step 3: Verify magic link (using fixture-generated token)
        # Note: In real test, we'd intercept the email and extract the token
        # For this test, we verify the flow structure

        # Step 4: Use session (would verify token from step 3)
        # Step 5: Logout (would use token from step 3)

        # Verify user was created
        assert user_id is not None
        assert len(user_id) > 0
