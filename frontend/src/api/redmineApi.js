/**
 * API layer for Redmine ticket lookups. The Test Plan Generator only
 * asks the user for a ticket ID (and an optional extra description) —
 * `redmineDescription` (a required field for `POST /generate`) comes
 * from the ticket's own description here, so the user never has to
 * retype text that already exists in Redmine.
 */
import { apiClient } from "./client.js";
import { ENDPOINTS } from "./endpoints.js";

export const redmineApi = {
  /** Full ticket detail (title, status, description, ...) for one Redmine ticket id. */
  getTicketDetail(ticketId) {
    return apiClient.get(ENDPOINTS.redmineTicketDebug(ticketId));
  },
};
