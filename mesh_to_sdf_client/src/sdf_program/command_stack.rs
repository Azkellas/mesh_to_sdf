use std::collections::VecDeque;

use web_time::Instant;

use super::{Parameters, Settings};

#[derive(Debug, Clone)]
pub struct State {
    pub parameters: Parameters,
    pub settings: Settings,
}

#[derive(Debug, Clone)]
pub struct Command {
    pub old_state: State,
    pub new_state: State,
}

/// A poor-man's command stack.
/// The label and timestamp are used to identify the command.
/// When pushing a new command, it is compared to the last one.
/// If the label are the same and the timestamp within a ten frame delta, the command is not pushed but updated.
/// This is to avoid pushing a command for each drag.
#[derive(Default)]
pub struct CommandStack {
    /// Maximum number of commands to keep in the stack.
    stack_size: usize,
    /// Stack of commands that can be redone.
    redo_stack: VecDeque<(&'static str, Instant, Command)>,
    // Stack of commands that can be undone.
    undo_stack: VecDeque<(&'static str, Instant, Command)>,
    /// Current transaction, if any.
    /// A transaction is a command that is being edited or might be.
    /// It is not yet in the undo stack.
    current_transaction: Option<(&'static str, Instant, Command)>,
}

impl CommandStack {
    /// Create a new command stack.
    pub fn new(stack_size: usize) -> Self {
        Self {
            stack_size,
            redo_stack: VecDeque::with_capacity(stack_size),
            undo_stack: VecDeque::with_capacity(stack_size),
            current_transaction: None,
        }
    }

    /// Push a new command to the stack.
    /// If a `current_transaction` is present:
    /// - if the label are the same and timestamp is within 10 deltas, the transaction is updated.
    /// - otherwise, the transaction is pushed to the undo stack.
    ///   and the current transaction is set to the new command.
    pub fn push(&mut self, label: &'static str, command: Command) {
        if let Some(transaction) = self.current_transaction.as_mut() {
            if transaction.0 == label && transaction.1.elapsed().as_secs_f32() < 10.0 / 60.0 {
                transaction.2.new_state = command.new_state;
                return;
            } else {
                self.redo_stack.clear();
                self.undo_stack.push_back(transaction.clone());
                self.current_transaction = None;
            }
        }

        self.current_transaction = Some((label, Instant::now(), command));
    }

    /// Undo the last command. Returns the command that was undone if any.
    pub fn undo(&mut self) -> Option<Command> {
        if self.current_transaction.is_some() {
            self.redo_stack.clear();
            self.undo_stack
                .push_back(self.current_transaction.take().unwrap());
            self.current_transaction = None;
        }

        if let Some(command) = self.undo_stack.pop_back() {
            self.redo_stack.push_back(command.clone());
            Some(command.2)
        } else {
            None
        }
    }

    /// Redo the last command. Returns the command that was redone if any.
    pub fn redo(&mut self) -> Option<Command> {
        if self.current_transaction.is_some() {
            self.redo_stack.clear();
            self.undo_stack
                .push_back(self.current_transaction.take().unwrap());
            self.current_transaction = None;
        }

        if let Some(command) = self.redo_stack.pop_back() {
            self.undo_stack.push_back(command.clone());
            Some(command.2)
        } else {
            None
        }
    }
}
