# Statement of Work — Freight Data Services

**This document is entirely fictional.** Every party, figure, date and clause was written
for this repository to exercise the section rules. It is not derived from any real
agreement. Its only job is to give you something to run `run.py` against before you point
it at a document of your own.

It deliberately includes the shapes the rules exist to handle: ALL-CAPS headings, Title
Case numbered clauses, a heading with no body of its own, decimal subsections, a section
long enough to need splitting, and a heading glued to the paragraph after it.

## 1. INTRODUCTION AND OVERVIEW

This Statement of Work (the "SOW") between Cedar Analytics Ltd ("Supplier") and Northwind
Logistics, Inc. ("Northwind") is subject to the terms of the Master Services Agreement (the
"MSA") dated 14 March 2023. Where this SOW and the MSA conflict, the MSA governs. This SOW
takes effect on the date both parties have signed it.

## 2. DEFINITIONS

"Change Request" means a request for the addition, removal or modification of this SOW or
any of its Attachments. "Freight Data Set" means the consignment records described in
Attachment 1. "Service Window" means 07:00 to 19:00 UTC on Business Days.

## 3. Project Location

3.1. The work on the Project will be performed at Supplier's facilities.

## 4. Technology

4.1. Northwind Applications

Supplier will use the following applications provided by Northwind, or by third parties on
behalf of Northwind, for performance of the Service: the consignment tracking platform, the
tariff engine, and the reporting warehouse. Northwind will provide access to each of these
and the required licences prior to knowledge transfer. Any share drive access method and
supporting protocols will align to Northwind's current security standards. Supplier may not
extract the Freight Data Set to any environment not listed in Attachment 1.

4.2. Northwind Hardware

Supplier does not anticipate any new equipment requirements for Northwind. Existing
equipment may need to be reconfigured to accommodate the Services defined in this SOW.

4.3. Supporting Requirements

Northwind will provide two named administrators and will maintain the tariff engine at a
supported version for the duration of the Services. Supplier shall notify Northwind of any
dependency that is not met within five Business Days of becoming aware of it.

## 5. TIMELINE

5.1. Services duration is 3 years from the Effective Date of this SOW. Unless Northwind
gives written notice of non-renewal at least 90 days before the end of the term, the term
renews automatically until 14 March 2029.

5.2. Transition runs from January 2024 to June 2024, with managed services beginning
1 July 2024.

## 6. PRICING AND PAYMENT

Northwind will pay Supplier for the Services in accordance with the monthly fees listed
below. Northwind's financial year starts on 1 May and ends on 30 April each year. Invoices
issued by Supplier will be paid in accordance with the terms of the MSA.

Estimated fees for the full term are $1,480,000, of which $310,000 is transition fees.
Year one is $520,000, year two $500,000 and year three $460,000. Expenses are reimbursable
up to $4,000 in aggregate, and any single expense over $500 requires Northwind's prior
written approval. All charges are subject to adjustment if the applicable exchange rate
moves by 3% or more from the rate in effect on the Effective Date. Any change to the fees
must be made by written Change Order.

## 7. Attachment 1 - Data Scope

## 7.1. Freight Data Set

The Freight Data Set comprises consignment records for the regions listed in Table 1, being
Northern Europe, Iberia and the Nordics. Supplier will process approximately 90,000 records
per month. Records are delivered nightly as compressed CSV to the reporting warehouse, and
Supplier shall confirm receipt within one hour of each delivery. Where a nightly delivery
fails, Supplier must raise an incident before 09:00 UTC and provide a root cause within
three Business Days. Northwind may withhold a delivery where it reasonably believes the
records contain personal data outside the agreed schema, and in that event the parties will
agree a remediation plan within ten Business Days. Supplier is not permitted to retain any
record for longer than the retention period stated in Attachment 2, and must certify
deletion annually. Supplier shall maintain an audit log of every access to the Freight Data
Set, retain that log for 24 months, and make it available to Northwind on request. Where
Supplier engages a subcontractor to process any part of the Freight Data Set, Supplier
remains responsible for that subcontractor's performance and must obtain Northwind's prior
written consent. Supplier will report monthly on volume, error rate and mean processing
time, and will meet a processing accuracy target of 99.5% measured monthly. If accuracy
falls below 99.0% in any two consecutive months, Northwind may terminate this SOW on 30
days' notice without termination charges. Supplier shall not vary the schema without a
Change Request approved by both parties. The parties will review the schema every six
months, and either party may propose changes at that review. Supplier must encrypt the
Freight Data Set at rest and in transit using the standards set out in Attachment 3, and
must notify Northwind within 24 hours of becoming aware of any unauthorised access.

## 8. TERMINATION CHARGES

8.1 If Northwind terminates this SOW without cause, the total allowable termination charge
is $92,000 in year one, $61,000 in year two and $0 thereafter.

## 9. SIGNATURES IN WITNESS WHEREOF the Parties have executed this SOW as of the dates set out below, each by its duly authorised representative.
