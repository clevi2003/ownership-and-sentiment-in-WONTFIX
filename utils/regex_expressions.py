import re

# https://docs.python.org/3/library/re.html since it's impossible to remember them all

# leave this alone lest you fall into regex hell
# also, this looks for patterns like "fixes #123" or "closes: #456" or "closed #789" or similar in PR bodies
# basically any format that I could think of, which is a lot
CLOSING_REFERENCE_PATTERN = re.compile(r"""(?ix)\b(?:close|closes|closed|closing|
                                                            fix|fixes|fixed|fixing|
                                                            resolve|resolves|resolved|resolving)
                                        \b[\s:,-]*(?:issue\s+)?(?P<ref>\#\d+|gh-\d+|[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\#\d+|
                                                                https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/issues/\d+)""",
                                        re.VERBOSE)
# CLOSING_CLAUSE_PATTERN = re.compile(r"""(?ix)\b(?:close|closes|closed|closing|
#                                                 fix|fixes|fixed|fixing|
#                                                 resolve|resolves|resolved|resolving)\b(?P<tail>
#                                                                                        [^\n.;]*)""", re.VERBOSE)
CLOSING_CLAUSE_PATTERN = re.compile(r"""(?ix)\b(?:close|closes|closed|closing|
                                                       fix|fixes|fixed|fixing|
                                                       resolve|resolves|resolved|resolving)\b(?P<tail>[^\n.;]*)""",
                                    re.VERBOSE)
ISSUE_REF_PATTERN = re.compile(r"""(?ix)(?:
                                                \#(?P<plain>\d+)
                                                |
                                                gh-(?P<gh>\d+)
                                                |
                                                [A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\#(?P<repo>\d+)
                                                |
                                                https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/issues/(?P<url>\d+))""",
                                re.VERBOSE)
ISSUE_REF_IN_CLAUSE_PATTERN = re.compile(r"""(?ix)(?:
                                                        \#(?P<plain>\d+)
                                                        |
                                                        gh-(?P<gh>\d+)
                                                        |
                                                        [A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\#(?P<repo>\d+)
                                                        |
                                                        https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/issues/(?P<url>\d+)
                                                    )""", re.VERBOSE)
COMMIT_ISSUE_REF_PATTERN = re.compile(r"""(?ix)(?:
                                                        \#(?P<plain>\d+)
                                                        |
                                                        gh-(?P<gh>\d+)
                                                        |
                                                        [A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\#(?P<repo>\d+)
                                                        |
                                                        https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/issues/(?P<url>\d+)
                                                    )""", re.VERBOSE)
ISSUE_NUMBER_FROM_REF_PATTERNS = [
    re.compile(r"(?i)^#(?P<num>\d+)$"),
    re.compile(r"(?i)^gh-(?P<num>\d+)$"),
    re.compile(r"(?i)^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+#(?P<num>\d+)$"),
    re.compile(r"(?i)^https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/issues/(?P<num>\d+)$")]
# this is less coverage and misses some things but still leave alone in case we revert back
# CLOSING_KEYWORD_PATTERN = re.compile(r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)\b", flags=re.IGNORECASE,)
# ISSUE_REF_PATTERN = re.compile(r"(?<![A-Za-z0-9_-])#(\d+)\b")