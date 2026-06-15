"""Concrete :class:`DataPlaneClient` adapters.

``noop`` is an in-memory reference/CI adapter (no external deps beyond
tensordict); ``transfer_queue`` is the production TransferQueue-backed
adapter. Adapters are imported lazily by the factory so importing the
package never forces an unused backend's dependencies.
"""
