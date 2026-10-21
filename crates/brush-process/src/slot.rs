use std::sync::Arc;

use tokio::sync::watch;

/// Read-only frame-indexed view of splat snapshots published by a
/// single producer (the train / load stream) into a [`SlotSender`].
#[derive(Clone)]
pub struct Slot<T> {
    rx: watch::Receiver<Vec<T>>,
    tx: Option<Arc<watch::Sender<Vec<T>>>>,
}

impl<T: Clone + Send + Sync + 'static> Slot<T> {
    /// An empty, never-updated `Slot`. Useful as a placeholder before
    /// a process has been wired up.
    pub fn empty() -> Self {
        // Drop the sender immediately — the receiver still works for
        // reads but observes the initial empty Vec forever.
        let (_, rx) = watch::channel(Vec::new());
        Self { rx, tx: None }
    }

    pub fn get(&self, index: usize) -> Option<T> {
        self.rx.borrow().get(index).cloned()
    }

    pub fn latest(&self) -> Option<T> {
        self.rx.borrow().last().cloned()
    }

    pub fn len(&self) -> usize {
        self.rx.borrow().len()
    }

    pub fn is_empty(&self) -> bool {
        self.rx.borrow().is_empty()
    }

    /// Replace the existing value at `index` from the consumer side, e.g. to
    /// swap in a reduced-detail copy of a loaded scene so the full one can be
    /// freed. Returns false (and does nothing) if there is no value at `index`.
    pub fn replace(&self, index: usize, value: T) -> bool {
        let Some(tx) = &self.tx else {
            return false;
        };
        tx.send_if_modified(|vec| match vec.get_mut(index) {
            Some(slot) => {
                *slot = value;
                true
            }
            None => false,
        })
    }
}

impl<T: Clone + Send + Sync + 'static> Default for Slot<T> {
    fn default() -> Self {
        Self::empty()
    }
}

/// Write side of a [`Slot`]. Single producer; consumers hold cloneable
/// [`Slot`] receivers.
pub struct SlotSender<T> {
    tx: Arc<watch::Sender<Vec<T>>>,
}

impl<T: Send + Sync + 'static> SlotSender<T> {
    /// Replace value at `index`. If `index == len()` the value is
    /// appended. Panics if `index > len()`.
    pub fn set(&self, index: usize, value: T) {
        self.tx.send_modify(|vec| match index.cmp(&vec.len()) {
            std::cmp::Ordering::Less => vec[index] = value,
            std::cmp::Ordering::Equal => vec.push(value),
            std::cmp::Ordering::Greater => {
                panic!(
                    "SlotSender::set index {index} past end (len = {})",
                    vec.len()
                )
            }
        });
    }

    pub fn clear(&self) {
        self.tx.send_modify(|vec| vec.clear());
    }
}

/// Build a fresh `(SlotSender, Slot)` pair backed by a single
/// `watch::channel`. The sender goes to the producer; the receiver
/// (and any clones) goes to consumers.
pub fn channel<T: Send + Sync + 'static>() -> (SlotSender<T>, Slot<T>) {
    let (tx, rx) = watch::channel(Vec::new());
    let tx = Arc::new(tx);
    (
        SlotSender { tx: tx.clone() },
        Slot { rx, tx: Some(tx) },
    )
}
