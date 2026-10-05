# ARC mail diagnostic — 7 September 2026

Update, 14 September 2026: the user confirmed that the Slurm emails arrived.
Delivery is now confirmed by the recipient. The earlier delay's cause and
the exact set of messages received have not been established. No further
notification test is needed on the current evidence.

Test job `8758069` on `htc` completed successfully (`0:0`), running from
12:44:27 to 12:44:47 BST. Its effective Slurm record, captured inside the
running job, contained:

```text
MailUser=carlo.palazzi@eng.ox.ac.uk MailType=BEGIN,END,FAIL
```

On 7 September the user reported that neither notification had arrived. Missing
submission flags therefore do not explain that initial nonreceipt. The scheduler
reports `MailProg=/bin/mail`, `MailDomain=(null)` and controller `htc-slurm`.
These settings do not establish successful hand-off or mailbox delivery.
Controller mail logs and recipient mail tracing are not exposed by the normal
job-accounting commands, so the delivery failure's exact cause is unresolved.

For ARC support, supply the job ID, cluster, recipient and times above and ask
whether Slurm invoked the mail handler, whether the relay accepted the messages,
and whether the recipient domain rejected or quarantined them. Check mailbox
junk/quarantine in parallel. No recipient address has been changed or test
message sent outside the explicitly requested Slurm notification workflow.

The reconciliation submission wrapper now explicitly supplies the recipient
and BEGIN/END/FAIL policy to both worker and QA jobs and records the effective
scheduler settings. Job status and result QA remain the authoritative progress
checks; email receipt is not assumed.
