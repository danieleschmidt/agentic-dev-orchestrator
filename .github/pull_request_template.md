# Pull Request

## 📋 Summary
<!-- Provide a brief description of the changes in this PR -->

**Type of Change:**
- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🔧 Configuration change
- [ ] 🧹 Code cleanup/refactoring
- [ ] 🔒 Security improvement
- [ ] ⚡ Performance improvement
- [ ] 🧪 Test improvement

## 🎯 Related Issues
<!-- Link related issues using keywords like "Closes #123" or "Fixes #456" -->
- Closes #
- Related to #

## 🔄 Changes Made
<!-- Describe the changes made in this PR -->

### Added
- 

### Changed
- 

### Removed
- 

### Fixed
- 

## 🧪 Testing
<!-- Describe the testing performed to verify your changes -->

### Test Coverage
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] End-to-end tests added/updated
- [ ] Manual testing completed

### Test Commands
```bash
# Commands used to test the changes
pytest tests/
python -m ado --help
```

### Test Results
- [ ] All existing tests pass
- [ ] New tests pass
- [ ] Code coverage maintained/improved
- [ ] Performance benchmarks acceptable

## 📖 Documentation
<!-- Documentation changes and updates -->

- [ ] README.md updated
- [ ] API documentation updated
- [ ] Code comments added/updated
- [ ] Architecture docs updated
- [ ] Changelog entry added
- [ ] Migration guide provided (if breaking change)

## 🔒 Security
<!-- Security considerations and checklist -->

- [ ] No sensitive data exposed
- [ ] Input validation implemented
- [ ] Authentication/authorization considered
- [ ] Dependencies security-scanned
- [ ] Configuration securely handled
- [ ] Error messages don't leak information

## 📊 Performance Impact
<!-- Performance implications of the changes -->

**Expected Impact:**
- [ ] No performance impact
- [ ] Performance improvement
- [ ] Minor performance regression (justified)
- [ ] Significant performance change (benchmarked)

**Benchmarks (if applicable):**
```
Before: [metrics]
After:  [metrics]
```

## 🔍 Code Quality
<!-- Code quality checklist -->

- [ ] Code follows project style guidelines
- [ ] Self-review of code completed
- [ ] Code is self-documenting with clear variable names
- [ ] Complex logic is commented
- [ ] No debugging code or print statements left
- [ ] Error handling implemented appropriately

## 🚀 Deployment Considerations
<!-- Deployment and rollout considerations -->

- [ ] Database migrations (if applicable)
- [ ] Configuration changes documented
- [ ] Backward compatibility maintained
- [ ] Feature flags used (if applicable)
- [ ] Rollback plan documented

## ✅ Pre-merge Checklist
<!-- Final checklist before merge -->

### Code Quality
- [ ] Pre-commit hooks pass
- [ ] CI/CD pipeline passes
- [ ] Code review completed
- [ ] All conversations resolved

### Functionality
- [ ] Feature works as expected
- [ ] Edge cases handled
- [ ] Error conditions tested
- [ ] User experience considered

### Documentation
- [ ] Code is documented
- [ ] User-facing changes documented
- [ ] API changes documented
- [ ] Breaking changes highlighted

### Security & Compliance
- [ ] Security review completed (if needed)
- [ ] Dependencies are up to date
- [ ] No security vulnerabilities introduced
- [ ] Compliance requirements met

## 🎭 Screenshots/Demo
<!-- Include screenshots, GIFs, or demo links for UI changes -->

**Before:**
<!-- Screenshot or description of current state -->

**After:**
<!-- Screenshot or description of new state -->

## 🤔 Questions for Reviewers
<!-- Specific questions or concerns for reviewers -->

- 
- 

## 📝 Additional Notes
<!-- Any additional information for reviewers -->

## 🏷️ Labels
<!-- Suggest appropriate labels for this PR -->
**Suggested Labels:**
- Component: `component:cli`, `component:agents`, `component:backlog`
- Priority: `priority:low`, `priority:medium`, `priority:high`
- Size: `size:small`, `size:medium`, `size:large`
- Type: `type:bugfix`, `type:feature`, `type:enhancement`

---

## 📋 Review Guidelines for Maintainers

### Review Focus Areas
1. **Functionality**: Does the code work as intended?
2. **Code Quality**: Is the code clean, readable, and maintainable?
3. **Testing**: Are there adequate tests covering the changes?
4. **Documentation**: Is the code and functionality properly documented?
5. **Security**: Are there any security implications?
6. **Performance**: Are there any performance implications?
7. **Breaking Changes**: Are breaking changes properly communicated?

### Approval Criteria
- [ ] Code review completed by at least one maintainer
- [ ] All CI/CD checks pass
- [ ] Documentation is adequate
- [ ] Test coverage is maintained or improved
- [ ] Security considerations addressed
- [ ] Breaking changes are justified and documented