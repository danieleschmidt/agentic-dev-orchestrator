"""Integration tests for GitHub API interactions."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from autonomous_executor import AutonomousExecutor
from backlog_manager import BacklogManager, BacklogItem


@pytest.mark.integration
@pytest.mark.github
class TestGitHubIntegration:
    """Test GitHub API integration functionality."""

    def test_create_pull_request_integration(self, temp_workspace, mock_github_api, sample_backlog_item):
        """Test creating a pull request through GitHub API."""
        # Setup
        executor = AutonomousExecutor(str(temp_workspace))
        
        # Mock execution result
        execution_result = {
            "success": True,
            "item_id": sample_backlog_item.id,
            "artifacts": ["implementation.py", "test_implementation.py"],
            "branch_name": "feature/test-implementation",
            "commit_message": "feat: implement test feature\n\nImplement test feature as specified",
        }
        
        # Execute
        with patch.object(executor, '_execute_agent_pipeline', return_value=execution_result):
            result = executor.create_pull_request(sample_backlog_item, execution_result)
        
        # Verify
        assert result["success"] is True
        assert "pull_request_url" in result
        assert result["pull_request_number"] == 123
        mock_github_api.return_value.get_repo.assert_called_once()

    @pytest.mark.slow
    def test_github_repository_validation(self, temp_workspace, mock_github_api):
        """Test GitHub repository validation."""
        executor = AutonomousExecutor(str(temp_workspace))
        
        # Test valid repository
        mock_repo = Mock()
        mock_repo.permissions.push = True
        mock_github_api.return_value.get_repo.return_value = mock_repo
        
        result = executor.validate_github_repository("test/repo")
        assert result["valid"] is True
        assert result["permissions"]["push"] is True

    def test_github_branch_operations(self, temp_workspace, mock_github_api, sample_backlog_item):
        """Test GitHub branch creation and management."""
        executor = AutonomousExecutor(str(temp_workspace))
        
        # Mock repository and branch operations
        mock_repo = Mock()
        mock_branch = Mock()
        mock_branch.name = "main"
        mock_repo.get_branch.return_value = mock_branch
        mock_repo.create_git_ref.return_value = Mock()
        mock_github_api.return_value.get_repo.return_value = mock_repo
        
        # Test branch creation
        branch_name = f"feature/{sample_backlog_item.id}"
        result = executor.create_feature_branch(sample_backlog_item, "main")
        
        assert result["success"] is True
        assert result["branch_name"] == branch_name
        mock_repo.create_git_ref.assert_called_once()

    def test_github_pr_status_tracking(self, temp_workspace, mock_github_api, sample_backlog_item):
        """Test tracking pull request status changes."""
        executor = AutonomousExecutor(str(temp_workspace))
        
        # Mock PR with different states
        mock_pr = Mock()
        mock_pr.number = 123
        mock_pr.state = "open"
        mock_pr.merged = False
        mock_pr.html_url = "https://github.com/test/repo/pull/123"
        
        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github_api.return_value.get_repo.return_value = mock_repo
        
        # Test status tracking
        status = executor.track_pr_status(123)
        
        assert status["number"] == 123
        assert status["state"] == "open"
        assert status["merged"] is False
        assert status["url"] == "https://github.com/test/repo/pull/123"

    @pytest.mark.requires_network
    def test_github_api_rate_limiting(self, temp_workspace):
        """Test GitHub API rate limiting handling."""
        # This test would check actual rate limiting with real API
        # Skip in CI/CD unless explicitly enabled
        pytest.skip("Requires real GitHub API access for rate limiting test")

    def test_github_webhook_handling(self, temp_workspace, mock_github_api):
        """Test handling GitHub webhook events."""
        executor = AutonomousExecutor(str(temp_workspace))
        
        # Mock webhook payload
        webhook_payload = {
            "action": "closed",
            "pull_request": {
                "number": 123,
                "merged": True,
                "head": {
                    "ref": "feature/test-001"
                }
            }
        }
        
        # Test webhook processing
        result = executor.process_github_webhook(webhook_payload)
        
        assert result["processed"] is True
        assert result["action"] == "pull_request_merged"
        assert result["pr_number"] == 123

    def test_github_repository_permissions(self, temp_workspace, mock_github_api):
        """Test checking GitHub repository permissions."""
        executor = AutonomousExecutor(str(temp_workspace))
        
        # Mock different permission scenarios
        scenarios = [
            {"admin": True, "push": True, "pull": True},
            {"admin": False, "push": True, "pull": True},
            {"admin": False, "push": False, "pull": True},
        ]
        
        for permissions in scenarios:
            mock_repo = Mock()
            mock_repo.permissions = Mock(**permissions)
            mock_github_api.return_value.get_repo.return_value = mock_repo
            
            result = executor.check_repository_permissions("test/repo")
            
            assert result["admin"] == permissions["admin"]
            assert result["push"] == permissions["push"]
            assert result["pull"] == permissions["pull"]

    def test_github_issue_linking(self, temp_workspace, mock_github_api, sample_backlog_item):
        """Test linking backlog items to GitHub issues."""
        executor = AutonomousExecutor(str(temp_workspace))
        
        # Mock GitHub issue
        mock_issue = Mock()
        mock_issue.number = 456
        mock_issue.title = sample_backlog_item.title
        mock_issue.state = "open"
        
        mock_repo = Mock()
        mock_repo.get_issue.return_value = mock_issue
        mock_github_api.return_value.get_repo.return_value = mock_repo
        
        # Test issue linking
        result = executor.link_to_github_issue(sample_backlog_item, 456)
        
        assert result["linked"] is True
        assert result["issue_number"] == 456
        assert result["issue_title"] == sample_backlog_item.title

    def test_github_commit_operations(self, temp_workspace, mock_github_api, sample_backlog_item):
        """Test GitHub commit creation and management."""
        executor = AutonomousExecutor(str(temp_workspace))
        
        # Mock file content and commit operations
        mock_repo = Mock()
        mock_file = Mock()
        mock_file.decoded_content = b"existing content"
        mock_repo.get_contents.return_value = mock_file
        mock_repo.update_file.return_value = {"commit": Mock(sha="abc123")}
        mock_github_api.return_value.get_repo.return_value = mock_repo
        
        # Test commit creation
        changes = {
            "file1.py": "new content for file1",
            "file2.py": "new content for file2"
        }
        
        result = executor.create_commits(sample_backlog_item, changes, "feature/test-branch")
        
        assert result["success"] is True
        assert len(result["commits"]) == len(changes)
        assert all("sha" in commit for commit in result["commits"])

    def test_github_error_handling(self, temp_workspace, mock_github_api):
        """Test GitHub API error handling."""
        executor = AutonomousExecutor(str(temp_workspace))
        
        # Mock GitHub API exceptions
        from github import GithubException
        
        mock_github_api.return_value.get_repo.side_effect = GithubException(404, "Not Found")
        
        result = executor.validate_github_repository("nonexistent/repo")
        
        assert result["valid"] is False
        assert result["error"]["status_code"] == 404
        assert "Not Found" in result["error"]["message"]