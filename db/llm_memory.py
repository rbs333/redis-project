class LLMMemoryLayer:
    """User memory accessor class"""

    def __init__(self, client):
        # DI approach with singleton maybe
        self.client = client
        self.index = client.ft("idx:users")
        self.key = "user"

    # Required methods
    def add_user_chat_msg(self, user_id: str, msg: str):
        """add conversation history to memory layer"""
        MAX_MESSAGES = 5
        path = "$.recentChatHistory"

        # Is there a redis equivalent to a SQL injection attack?
        num_msgs = self.client.json().arrappend(f"user:{user_id}", path, msg)[0]
        if num_msgs > MAX_MESSAGES:
            self.client.json().arrpop(f"{self.key}:{user_id}", path, 0)

    def add_user_relevant_info(self, user_id: str, relevantInfo: list[str]):
        """add conversation history to memory layer"""
        # Note: this could also work as a set potentially

        for info in relevantInfo:
            self.client.json().arrappend(
                f"{self.key}:{user_id}", "$.relevantInfo", info
            )[0]

    def fetch_user(self, user_id):
        """fetch user and conversation history from memory layer"""
        return self.client.json().get(f"{self.key}:{user_id}")

    def clear_user_context(self, user_id: str):
        """clear the memory layer"""
        for path in ["$.recentChatHistory", "$.relevantInfo"]:
            self.client.json().set(f"{self.key}:{user_id}", path, [])
