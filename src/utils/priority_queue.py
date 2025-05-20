"""
Priority queue implementation for batch processing.
"""

import heapq
import threading
from typing import Any, Optional, Tuple, List, Dict
from dataclasses import dataclass, field
import time
import json
import sqlite3


@dataclass(order=True)
class PriorityItem:
    """Item with priority for queue."""

    priority: int
    sequence: int  # For FIFO ordering within same priority
    data: Any = field(compare=False)
    added_time: float = field(default_factory=time.time, compare=False)


class ThreadSafePriorityQueue:
    """Thread-safe priority queue implementation."""

    def __init__(self):
        """Initialize priority queue."""
        self._queue = []
        self._lock = threading.Lock()
        self._sequence = 0
        self._condition = threading.Condition(self._lock)

    def put(self, item: Any, priority: int = 1):
        """Add item to queue with priority.

        Args:
            item: Item to add
            priority: Priority (lower number = higher priority)
        """
        with self._lock:
            self._sequence += 1
            priority_item = PriorityItem(
                priority=priority, sequence=self._sequence, data=item
            )
            heapq.heappush(self._queue, priority_item)
            self._condition.notify()

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[Any]:
        """Get item from queue.

        Args:
            block: Whether to block if queue is empty
            timeout: Timeout in seconds

        Returns:
            Item from queue or None if timeout
        """
        with self._lock:
            end_time = time.time() + timeout if timeout else None

            while not self._queue:
                if not block:
                    return None

                remaining = None
                if end_time:
                    remaining = end_time - time.time()
                    if remaining <= 0:
                        return None

                if not self._condition.wait(remaining):
                    return None  # Timeout

            item = heapq.heappop(self._queue)
            return item.data

    def peek(self) -> Optional[Any]:
        """Peek at next item without removing it.

        Returns:
            Next item or None if queue is empty
        """
        with self._lock:
            if self._queue:
                return self._queue[0].data
            return None

    def size(self) -> int:
        """Get queue size.

        Returns:
            Number of items in queue
        """
        with self._lock:
            return len(self._queue)

    def empty(self) -> bool:
        """Check if queue is empty.

        Returns:
            True if queue is empty
        """
        with self._lock:
            return len(self._queue) == 0

    def clear(self):
        """Clear all items from queue."""
        with self._lock:
            self._queue.clear()
            self._sequence = 0

    def get_all_priorities(self) -> List[Tuple[int, Any]]:
        """Get all items with their priorities.

        Returns:
            List of (priority, item) tuples
        """
        with self._lock:
            return [(item.priority, item.data) for item in self._queue]


class PersistentPriorityQueue:
    """Priority queue with SQLite persistence."""

    def __init__(self, db_path: str, table_name: str = "priority_queue"):
        """Initialize persistent priority queue.

        Args:
            db_path: Path to SQLite database
            table_name: Name of queue table
        """
        self.db_path = db_path
        self.table_name = table_name
        self._lock = threading.Lock()
        self._initialize_db()

    def _initialize_db(self):
        """Initialize database table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    priority INTEGER NOT NULL,
                    data TEXT NOT NULL,
                    added_time REAL NOT NULL,
                    metadata TEXT
                )
            """
            )

            # Create index for efficient priority ordering
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_priority
                ON {self.table_name}(priority, id)
            """
            )

    def put(self, item: Any, priority: int = 1, metadata: Optional[Dict] = None):
        """Add item to queue.

        Args:
            item: Item to add (will be JSON serialized)
            priority: Priority level
            metadata: Optional metadata
        """
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    f"""
                    INSERT INTO {self.table_name} (priority, data, added_time, metadata)
                    VALUES (?, ?, ?, ?)
                """,
                    (
                        priority,
                        json.dumps(item) if not isinstance(item, str) else item,
                        time.time(),
                        json.dumps(metadata) if metadata else None,
                    ),
                )

    def get(self) -> Optional[Tuple[int, Any]]:
        """Get highest priority item.

        Returns:
            Tuple of (id, item) or None if empty
        """
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    f"""
                    SELECT id, data FROM {self.table_name}
                    ORDER BY priority, id
                    LIMIT 1
                """
                )

                row = cursor.fetchone()
                if row:
                    # Remove from queue
                    conn.execute(
                        f"DELETE FROM {self.table_name} WHERE id = ?", (row[0],)
                    )

                    # Parse data
                    try:
                        data = json.loads(row[1])
                    except json.JSONDecodeError:
                        data = row[1]

                    return (row[0], data)

                return None

    def peek(self) -> Optional[Any]:
        """Peek at next item without removing.

        Returns:
            Next item or None if empty
        """
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    f"""
                    SELECT data FROM {self.table_name}
                    ORDER BY priority, id
                    LIMIT 1
                """
                )

                row = cursor.fetchone()
                if row:
                    try:
                        return json.loads(row[0])
                    except json.JSONDecodeError:
                        return row[0]

                return None

    def size(self) -> int:
        """Get queue size.

        Returns:
            Number of items in queue
        """
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                return cursor.fetchone()[0]

    def get_by_priority(self, priority: int, limit: int = 100) -> List[Tuple[int, Any]]:
        """Get items with specific priority.

        Args:
            priority: Priority level
            limit: Maximum items to return

        Returns:
            List of (id, item) tuples
        """
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    f"""
                    SELECT id, data FROM {self.table_name}
                    WHERE priority = ?
                    ORDER BY id
                    LIMIT ?
                """,
                    (priority, limit),
                )

                results = []
                for row in cursor.fetchall():
                    try:
                        data = json.loads(row[1])
                    except json.JSONDecodeError:
                        data = row[1]
                    results.append((row[0], data))

                return results

    def update_priority(self, item_id: int, new_priority: int):
        """Update priority of an item.

        Args:
            item_id: Item ID
            new_priority: New priority level
        """
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    f"""
                    UPDATE {self.table_name}
                    SET priority = ?
                    WHERE id = ?
                """,
                    (new_priority, item_id),
                )

    def clear(self):
        """Clear all items from queue."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(f"DELETE FROM {self.table_name}")

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics.

        Returns:
            Dictionary with queue statistics
        """
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                # Total count
                total = conn.execute(
                    f"SELECT COUNT(*) FROM {self.table_name}"
                ).fetchone()[0]

                # Count by priority
                priority_counts = {}
                cursor = conn.execute(
                    f"""
                    SELECT priority, COUNT(*)
                    FROM {self.table_name}
                    GROUP BY priority
                    ORDER BY priority
                """
                )

                for priority, count in cursor.fetchall():
                    priority_counts[priority] = count

                # Age of oldest item
                oldest = conn.execute(
                    f"""
                    SELECT MIN(added_time) FROM {self.table_name}
                """
                ).fetchone()[0]

                oldest_age = time.time() - oldest if oldest else 0

                return {
                    "total_items": total,
                    "priority_counts": priority_counts,
                    "oldest_item_age": oldest_age,
                }


class CircularBuffer:
    """Circular buffer for storing recent items."""

    def __init__(self, capacity: int):
        """Initialize circular buffer.

        Args:
            capacity: Maximum number of items
        """
        self.capacity = capacity
        self.buffer = []
        self.index = 0
        self._lock = threading.Lock()

    def add(self, item: Any):
        """Add item to buffer.

        Args:
            item: Item to add
        """
        with self._lock:
            if len(self.buffer) < self.capacity:
                self.buffer.append(item)
            else:
                self.buffer[self.index] = item
                self.index = (self.index + 1) % self.capacity

    def get_all(self) -> List[Any]:
        """Get all items in buffer.

        Returns:
            List of items in order
        """
        with self._lock:
            if len(self.buffer) < self.capacity:
                return self.buffer.copy()
            else:
                # Return items in correct order
                return self.buffer[self.index :] + self.buffer[: self.index]

    def clear(self):
        """Clear buffer."""
        with self._lock:
            self.buffer.clear()
            self.index = 0
